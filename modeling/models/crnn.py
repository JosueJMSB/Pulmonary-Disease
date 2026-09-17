"""Arquitectura CRNN (CNN + BiGRU + atencion temporal) para EPOC frente a control.

Este modulo solo define la red. El entrenamiento, la busqueda de
hiperparametros, los folds, la normalizacion, la agregacion por paciente y los
checkpoints son los de ``models/cnn.py``, que construye esta red cuando el TOML
declara ``[model] architecture = "crnn"``. Asi la CRNN sigue exactamente el
mismo protocolo que la CNN y la comparacion entre ambas es directa.

El frente convolucional esta ADAPTADO, no es identico al de la CNN: usa los
mismos bloques de dos Conv3x3 + BN + ReLU, pero el ultimo bloque tiene 128
canales (no 256) y el pooling reduce la frecuencia 16x y el tiempo solo 4x.
Con eso quedan 77 pasos temporales para la BiGRU.

Los 77 pasos estan separados unos 64 ms (4 tramas de 16 ms), pero cada uno
incorpora un contexto acustico mayor: el campo receptivo de las convoluciones
y, sobre todo, la BiGRU, que recorre el segmento completo en ambas
direcciones. Los pesos de atencion indican en que pasos se apoyo el modelo
para decidir; no son una localizacion clinica de crepitantes ni sibilancias.

Solo depende de torch: no importa ``models/cnn.py``, lo que evita imports
circulares.
"""

from __future__ import annotations

import math

import torch
from torch import nn

ARCHITECTURE_NAME = "CopdCRNN"


class CRNNConvBlock(nn.Sequential):
    """Conv3x3-BN-ReLU, Conv3x3-BN-ReLU, MaxPool (frecuencia, tiempo)."""

    def __init__(self, in_channels: int, out_channels: int, pool_size: tuple[int, int]):
        pool = (int(pool_size[0]), int(pool_size[1]))
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool, stride=pool),
        )


class TemporalAttention(nn.Module):
    """Atencion aditiva sobre el eje temporal.

    ``Linear(D, A) -> tanh -> Linear(A, 1) -> softmax`` sobre los T pasos, y
    suma de la secuencia ponderada por esos pesos. El softmax se calcula en
    float32 incluso con AMP, para que los pesos sumen 1 sin error de redondeo
    de float16.
    """

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(sequence).squeeze(-1).float()        # (B, T)
        weights = torch.softmax(scores, dim=1)                     # (B, T), suman 1
        pooled = torch.sum(weights.unsqueeze(-1).to(sequence.dtype) * sequence, dim=1)  # (B, D)
        return pooled, weights


class CopdCRNN(nn.Module):
    """CRNN: frente convolucional -> proyeccion -> BiGRU -> atencion -> cabeza.

    Con la configuracion de ``crnn.toml`` y entrada ``(B, 1, 64, 309)``:

    - frente:     (B, 128, 4, 77)
    - secuencia:  (B, 77, 512)
    - proyeccion: (B, 77, 128)
    - BiGRU:      (B, 77, 256)
    - atencion:   pesos (B, 77), vector (B, 256)
    - cabeza:     logit (B, 1)

    ``forward`` devuelve solo logits por defecto; con ``return_attention=True``
    devuelve ``(logits, pesos)``. Nunca aplica sigmoid: la probabilidad se
    calcula fuera, igual que en la CNN.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_mels: int = 64,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        pool_sizes: tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (2, 1), (2, 1)),
        projection_dim: int = 128,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.2,
        attention_dim: int = 64,
        head_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if len(channels) != len(pool_sizes):
            raise ValueError("channels y pool_sizes deben tener la misma longitud")
        frequency_reduction = math.prod(int(p[0]) for p in pool_sizes)
        if n_mels % frequency_reduction:
            raise ValueError(f"n_mels={n_mels} no es divisible por la reduccion en frecuencia {frequency_reduction}")
        self.n_frequency_out = n_mels // frequency_reduction

        blocks = []
        current = in_channels
        for out_channels, pool in zip(channels, pool_sizes):
            blocks.append(CRNNConvBlock(current, int(out_channels), pool))
            current = int(out_channels)
        self.features = nn.Sequential(*blocks)

        self.projection = nn.Sequential(
            nn.Linear(current * self.n_frequency_out, projection_dim, bias=False),
            nn.LayerNorm(projection_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(2 * gru_hidden, attention_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * gru_hidden, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    @staticmethod
    def _to_sequence(feature_map: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) -> (B, T, C*F): cada paso temporal concatena canales y bandas."""
        batch, channels, frequencies, steps = feature_map.shape
        return feature_map.permute(0, 3, 1, 2).reshape(batch, steps, channels * frequencies)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        sequence = self._to_sequence(self.features(x))
        projected = self.projection(sequence)
        # Tras .to(device) o load_state_dict, cuDNN puede recibir los pesos de la
        # GRU en bloques no contiguos: compactarlos evita copias y advertencias.
        self.gru.flatten_parameters()
        recurrent, _ = self.gru(projected)
        pooled, weights = self.attention(recurrent)
        logits = self.head(pooled)
        if return_attention:
            return logits, weights
        return logits

    @torch.no_grad()
    def intermediate_shapes(self, x: torch.Tensor) -> dict[str, tuple[int, ...]]:
        """Formas de cada etapa, en modo evaluacion y sin gradiente."""
        was_training = self.training
        self.eval()
        try:
            feature_map = self.features(x)
            sequence = self._to_sequence(feature_map)
            projected = self.projection(sequence)
            self.gru.flatten_parameters()
            recurrent, _ = self.gru(projected)
            pooled, weights = self.attention(recurrent)
            logits = self.head(pooled)
        finally:
            self.train(was_training)
        return {
            "features": tuple(feature_map.shape),
            "sequence": tuple(sequence.shape),
            "projection": tuple(projected.shape),
            "recurrent": tuple(recurrent.shape),
            "attention": tuple(weights.shape),
            "pooled": tuple(pooled.shape),
            "logits": tuple(logits.shape),
        }


# ---------------------------------------------------------------------------
# Construccion desde el TOML y desde un checkpoint
# ---------------------------------------------------------------------------

def _kwargs_from_config(cfg: dict) -> dict:
    c = cfg["crnn"]
    return {
        "in_channels": int(c.get("in_channels", 1)),
        "n_mels": int(cfg["logmel"]["n_mels"]),
        "channels": tuple(int(v) for v in c["channels"]),
        "pool_sizes": tuple((int(p[0]), int(p[1])) for p in c["pool_sizes"]),
        "projection_dim": int(c["projection_dim"]),
        "gru_hidden": int(c["gru_hidden"]),
        "gru_layers": int(c["gru_layers"]),
        "gru_dropout": float(c["gru_dropout"]),
        "attention_dim": int(c["attention_dim"]),
        "head_hidden": int(c["head_hidden"]),
    }


def build_crnn(cfg: dict, dropout: float) -> CopdCRNN:
    """``dropout`` es el de la cabeza: el hiperparametro que se busca. El
    dropout entre capas de la GRU es fijo (``[crnn] gru_dropout``)."""
    return CopdCRNN(**_kwargs_from_config(cfg), dropout=float(dropout))


def crnn_architecture_description(cfg: dict) -> dict:
    """Descripcion JSON-nativa, para huella de ejecucion, checkpoints y metadata."""
    kwargs = _kwargs_from_config(cfg)
    model = CopdCRNN(**kwargs, dropout=0.0)
    n_frames = int(cfg["acoustic"]["expected_frames"])
    time_reduction = math.prod(p[1] for p in kwargs["pool_sizes"])
    frame_ms = 1000.0 * int(cfg["acoustic"]["hop_length"]) / int(cfg["acoustic"]["sample_rate"])
    return {
        "class": ARCHITECTURE_NAME,
        "in_channels": kwargs["in_channels"],
        "channels": list(kwargs["channels"]),
        "pool_sizes": [list(p) for p in kwargs["pool_sizes"]],
        "projection_dim": kwargs["projection_dim"],
        "gru_hidden": kwargs["gru_hidden"],
        "gru_layers": kwargs["gru_layers"],
        "gru_dropout": kwargs["gru_dropout"],
        "attention_dim": kwargs["attention_dim"],
        "head_hidden": kwargs["head_hidden"],
        "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "input_shape": [kwargs["in_channels"], kwargs["n_mels"], n_frames],
        "sequence_steps": n_frames // time_reduction,
        "step_ms": frame_ms * time_reduction,
        "recurrent_output_dim": 2 * kwargs["gru_hidden"],
        "note": "pesos de atencion = en que pasos se apoyo el modelo; no localizacion clinica",
    }


def model_from_architecture(architecture: dict) -> CopdCRNN:
    """Reconstruye la red a partir de ``checkpoint["architecture"]``."""
    return CopdCRNN(
        in_channels=int(architecture["in_channels"]),
        n_mels=int(architecture["input_shape"][1]),
        channels=tuple(int(v) for v in architecture["channels"]),
        pool_sizes=tuple((int(p[0]), int(p[1])) for p in architecture["pool_sizes"]),
        projection_dim=int(architecture["projection_dim"]),
        gru_hidden=int(architecture["gru_hidden"]),
        gru_layers=int(architecture["gru_layers"]),
        gru_dropout=float(architecture["gru_dropout"]),
        attention_dim=int(architecture["attention_dim"]),
        head_hidden=int(architecture["head_hidden"]),
        dropout=float(architecture["dropout"]),
    )
