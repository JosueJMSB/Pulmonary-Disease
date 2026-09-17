"""CNN 2D propia sobre Log-Mel (1 x 64 x 309) para COPD frente a control.

Contiene todo lo especifico de la red:

- la arquitectura (``CopdCNN``, 1 205 921 parametros entrenables);
- la normalizacion por banda Mel, calculada solo con el conjunto que se entrena;
- SpecAugment reproducible, aplicado solo en entrenamiento;
- el bucle de entrenamiento con perdida ponderada, AMP y parada temprana;
- el protocolo de un fold: busqueda de (lr, dropout) en validation, reajuste
  sobre train + validation durante la mejor epoca y una unica evaluacion de test.

La agregacion segmento -> grabacion -> paciente, las metricas y la linea base
Dummy son las mismas funciones que usa la SVM (``evaluation.py`` y
``svm_rbf.compute_dummy_baseline``), para que la comparacion sea directa.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .. import data as dmod
from .. import evaluation as ev
from .. import splits as sp
from .svm_rbf import compute_dummy_baseline

ARCHITECTURE_NAME = "CopdCNN"
CHECKPOINT_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------------

def _conv_bn_relu(in_channels: int, out_channels: int) -> list[nn.Module]:
    # Sin bias: el BatchNorm que sigue ya aporta el desplazamiento (beta).
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class ConvBlock(nn.Sequential):
    """Conv-BN-ReLU, Conv-BN-ReLU, MaxPool 2x2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            *_conv_bn_relu(in_channels, out_channels),
            *_conv_bn_relu(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2),
        )


class CopdCNN(nn.Module):
    """CNN de 4 bloques convolucionales + cabeza densa, salida de 1 logit.

    Con ``channels=(32, 64, 128, 256)`` y ``hidden_units=128``:
    1 172 640 parametros convolucionales (incluidas sus BN) + 33 152 de
    Dense 256->128 con bias y su BN + 129 de Dense 128->1 = 1 205 921.

    El pooling global es ``x.mean(dim=(2, 3))``: matematicamente identico a
    ``AdaptiveAvgPool2d(1)``, pero su gradiente en CUDA es determinista,
    mientras que el de ``adaptive_avg_pool2d`` no lo es.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        hidden_units: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        blocks = []
        current = in_channels
        for out_channels in channels:
            blocks.append(ConvBlock(current, out_channels))
            current = out_channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(current, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


def build_model(cfg: dict, dropout: float) -> nn.Module:
    """Red del experimento segun ``[model] architecture`` del TOML.

    Todo el protocolo de este modulo (entrenamiento, folds, normalizacion,
    checkpoints) es compartido por la CNN y la CRNN: solo este punto, la
    descripcion de arquitectura y la reconstruccion desde checkpoint cambian.
    Sin la seccion [model] (cnn.toml) se construye la CNN.
    """
    if dmod.model_architecture(cfg) == "crnn":
        from . import crnn as crnn_arch

        return crnn_arch.build_crnn(cfg, dropout)
    model_cfg = cfg["cnn"]
    return CopdCNN(
        in_channels=int(model_cfg.get("in_channels", 1)),
        channels=tuple(int(c) for c in model_cfg["channels"]),
        hidden_units=int(model_cfg["hidden_units"]),
        dropout=float(dropout),
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def expected_parameters(cfg: dict) -> int:
    """``expected_parameters`` de la seccion de la red activa ([cnn] o [crnn])."""
    return int(cfg[dmod.model_architecture(cfg)]["expected_parameters"])


def architecture_description(cfg: dict) -> dict:
    """Descripcion JSON-nativa de la arquitectura, para huella y metadata."""
    if dmod.model_architecture(cfg) == "crnn":
        from . import crnn as crnn_arch

        return crnn_arch.crnn_architecture_description(cfg)
    model_cfg = cfg["cnn"]
    return {
        "class": ARCHITECTURE_NAME,
        "in_channels": int(model_cfg.get("in_channels", 1)),
        "channels": [int(c) for c in model_cfg["channels"]],
        "hidden_units": int(model_cfg["hidden_units"]),
        "kernel_size": 3,
        "pooling": "maxpool2x2 por bloque + media global",
        "n_parameters": count_parameters(build_model(cfg, dropout=0.0)),
        "input_shape": [
            int(model_cfg.get("in_channels", 1)),
            int(cfg["logmel"]["n_mels"]),
            int(cfg["acoustic"]["expected_frames"]),
        ],
    }


# ---------------------------------------------------------------------------
# Dispositivo, determinismo y semillas
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> torch.device:
    """``auto`` usa CUDA si esta disponible; ``cuda``/``cuda:N`` fallan con
    un error explicito si esa GPU no existe, en vez de caer a CPU en silencio."""
    if requested == "auto":
        return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("se pidio CUDA pero torch.cuda.is_available() es False")
        device = torch.device(requested)
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda:{index} no existe; hay {torch.cuda.device_count()} GPU visibles")
        return torch.device("cuda", index)
    raise ValueError(f"dispositivo no reconocido: {requested}")


def device_description(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return f"{device} {props.name}, {free / 1024**3:.1f} GiB libres de {total / 1024**3:.1f} GiB"


def configure_determinism(cfg: dict) -> None:
    """Debe llamarse antes de la primera operacion CUDA: cuBLAS lee
    ``CUBLAS_WORKSPACE_CONFIG`` al crear su handle."""
    det_cfg = cfg.get("determinism", {})
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = bool(det_cfg.get("cudnn_deterministic", True))
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        bool(det_cfg.get("deterministic_algorithms", True)),
        warn_only=bool(det_cfg.get("warn_only", True)),
    )


def torch_environment(device: torch.device) -> dict:
    cudnn_available = torch.backends.cudnn.is_available()
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version() if cudnn_available else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update({
            "gpu_name": props.name,
            "gpu_capability": f"{props.major}.{props.minor}",
            "gpu_total_memory_gib": round(props.total_memory / 1024**3, 2),
        })
    return info


def derive_seed(base: int, *parts) -> int:
    """Semilla estable derivada de ``base`` y de un contexto (dataset, fold,
    configuracion, fase...). La condicion NO forma parte del contexto: no_dn,
    dn y la version con augmentation parten de la misma inicializacion en cada
    fold/configuracion, lo que reduce ruido en las comparaciones emparejadas."""
    payload = ":".join(str(p) for p in (base, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big") % (2**31 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.utils.data.get_worker_info().seed % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Normalizacion por banda Mel (solo con el conjunto que se entrena)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray  # (n_mels,)
    std: np.ndarray   # (n_mels,)
    n_segments: int
    weighted: bool

    def to_dict(self) -> dict:
        return {
            "axis": "mel_band",
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "n_segments": int(self.n_segments),
            "weighted": bool(self.weighted),
        }

    def to_tensors(self) -> dict:
        return {
            "axis": "mel_band",
            "mean": torch.tensor(self.mean, dtype=torch.float32),
            "std": torch.tensor(self.std, dtype=torch.float32),
            "n_segments": int(self.n_segments),
            "weighted": bool(self.weighted),
        }


def compute_normalization(
    logmel: np.ndarray,
    positions,
    weights=None,
    std_floor: float = 1e-6,
    chunk_size: int = 256,
) -> NormalizationStats:
    """Media y desviacion de cada banda Mel sobre ``logmel[positions]``.

    Cada segmento contribuye con su media temporal ponderada por ``weights``
    (o uniforme si es ``None``). Todos los segmentos tienen el mismo numero de
    tramas, asi que sin pesos equivale a la media sobre todos los valores de
    la banda. Nunca se le pasan filas de validation ni de test.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        raise ValueError("no hay segmentos para calcular la normalizacion")
    if weights is None:
        w = np.full(positions.size, 1.0 / positions.size)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != positions.shape or (w < 0).any() or w.sum() <= 0:
            raise ValueError("pesos de normalizacion invalidos")
        w = w / w.sum()

    order = np.argsort(positions, kind="stable")
    positions, w = positions[order], w[order]
    n_bands = int(logmel.shape[2])
    first = np.zeros(n_bands, dtype=np.float64)
    second = np.zeros(n_bands, dtype=np.float64)
    for start in range(0, positions.size, chunk_size):
        idx = positions[start:start + chunk_size]
        wc = w[start:start + chunk_size]
        block = np.asarray(logmel[idx], dtype=np.float64)[:, 0]  # (n, bandas, tramas)
        first += (wc[:, None] * block.mean(axis=2)).sum(axis=0)
        second += (wc[:, None] * (block ** 2).mean(axis=2)).sum(axis=0)

    std = np.maximum(np.sqrt(np.maximum(second - first ** 2, 0.0)), std_floor)
    if not (np.isfinite(first).all() and np.isfinite(std).all()):
        raise FloatingPointError("estadisticas de normalizacion no finitas")
    return NormalizationStats(mean=first, std=std, n_segments=int(positions.size), weighted=weights is not None)


# ---------------------------------------------------------------------------
# Dataset, DataLoader y SpecAugment
# ---------------------------------------------------------------------------

class LogmelDataset(Dataset):
    """Devuelve ``(x normalizado (1, bandas, tramas), y, peso, indice)``.

    La normalizacion ocurre aqui (en los workers); SpecAugment no: se aplica
    por lote en el proceso principal, con un generador propio, para que el
    resultado no dependa de ``--num-workers``.
    """

    def __init__(self, logmel: np.ndarray, positions, labels, weights, stats: NormalizationStats):
        self.logmel = logmel
        self.positions = np.asarray(positions, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.weights = (
            np.ones(len(self.positions), dtype=np.float32)
            if weights is None else np.asarray(weights, dtype=np.float32)
        )
        if not (len(self.positions) == len(self.labels) == len(self.weights)):
            raise ValueError("positions, labels y weights deben tener la misma longitud")
        self.mean = stats.mean.astype(np.float32).reshape(1, -1, 1)
        self.std = stats.std.astype(np.float32).reshape(1, -1, 1)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        x = (np.asarray(self.logmel[self.positions[i]], dtype=np.float32) - self.mean) / self.std
        return (
            torch.from_numpy(x),
            torch.tensor(self.labels[i], dtype=torch.float32),
            torch.tensor(self.weights[i], dtype=torch.float32),
            torch.tensor(i, dtype=torch.long),
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        generator=generator,
    )
    if num_workers > 0:
        kwargs.update(worker_init_fn=_seed_worker, persistent_workers=True)
    return DataLoader(dataset, **kwargs)


def _needs_drop_last(n_samples: int, batch_size: int) -> bool:
    """Un ultimo lote de tamano 1 rompe BatchNorm1d en modo train. Solo en ese
    caso se descarta, y como el orden se baraja cada epoca, el segmento que
    queda fuera cambia de epoca a epoca."""
    if n_samples < 2:
        raise ValueError("se necesitan al menos 2 segmentos para entrenar con BatchNorm")
    return n_samples % batch_size == 1


def spec_augment(x: torch.Tensor, aug_cfg: dict, generator: torch.Generator) -> torch.Tensor:
    """Una mascara temporal (hasta N tramas) y una frecuencial (hasta M bandas)
    por muestra, cada una con su probabilidad, rellenas con ``fill_value``.

    ``x`` es ``(B, 1, bandas, tramas)`` ya normalizado. Todos los numeros
    aleatorios se generan en CPU con ``generator`` -siempre la misma cantidad,
    se aplique o no la mascara-, asi que el resultado es reproducible e
    independiente del dispositivo y de los workers. No modifica ``x``.
    """
    batch, _, n_bands, n_frames = x.shape
    max_frames = min(int(aug_cfg["time_mask_max_frames"]), n_frames)
    max_bands = min(int(aug_cfg["freq_mask_max_bands"]), n_bands)

    apply_t = torch.rand(batch, generator=generator) < float(aug_cfg["time_mask_probability"])
    width_t = torch.randint(0, max_frames + 1, (batch,), generator=generator)
    start_t = (torch.rand(batch, generator=generator) * (n_frames - width_t + 1).float()).floor().long()

    apply_f = torch.rand(batch, generator=generator) < float(aug_cfg["freq_mask_probability"])
    width_f = torch.randint(0, max_bands + 1, (batch,), generator=generator)
    start_f = (torch.rand(batch, generator=generator) * (n_bands - width_f + 1).float()).floor().long()

    frames = torch.arange(n_frames)
    bands = torch.arange(n_bands)
    mask_t = apply_t[:, None] & (frames[None, :] >= start_t[:, None]) & (frames[None, :] < (start_t + width_t)[:, None])
    mask_f = apply_f[:, None] & (bands[None, :] >= start_f[:, None]) & (bands[None, :] < (start_f + width_f)[:, None])
    mask = mask_t[:, None, None, :] | mask_f[:, None, :, None]
    return x.masked_fill(mask.to(x.device, non_blocking=True), float(aug_cfg.get("fill_value", 0.0)))


# ---------------------------------------------------------------------------
# Perdida, entrenamiento e inferencia
# ---------------------------------------------------------------------------

def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Media de ``w_s * BCE_s`` en el lote. Los pesos ya estan normalizados a
    media 1 sobre el conjunto de entrenamiento: no se usa ``pos_weight`` ni
    ningun otro balanceo adicional. Se calcula en float32 aunque haya AMP."""
    per_sample = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    return (per_sample * weights.float()).mean()


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    eval_batch_size: int
    max_epochs: int
    min_epochs: int
    patience: int
    weight_decay: float
    grad_clip_norm: float
    t_max: int
    eta_min: float
    amp: bool
    threshold: float
    normalization_weighted: bool
    std_floor: float
    num_workers: int
    seed_base: int

    @classmethod
    def from_config(cls, cfg: dict, num_workers: int) -> "TrainingSettings":
        t, n = cfg["training"], cfg["normalization"]
        return cls(
            batch_size=int(t["batch_size"]),
            eval_batch_size=int(t.get("eval_batch_size", 256)),
            max_epochs=int(t["max_epochs"]),
            min_epochs=int(t["min_epochs"]),
            patience=int(t["patience"]),
            weight_decay=float(t["weight_decay"]),
            grad_clip_norm=float(t["grad_clip_norm"]),
            t_max=int(t["t_max"]),
            eta_min=float(t["eta_min"]),
            amp=bool(t["amp"]),
            threshold=float(cfg["evaluation"]["decision_threshold"]),
            normalization_weighted=bool(n["weighted"]),
            std_floor=float(n["std_floor"]),
            num_workers=int(num_workers),
            seed_base=int(cfg["seeds"]["base"]),
        )

    def use_amp(self, device: torch.device) -> bool:
        return self.amp and device.type == "cuda"


@dataclass(frozen=True)
class SearchConfiguration:
    index: int
    lr: float
    dropout: float


def search_configurations(cfg: dict) -> list[SearchConfiguration]:
    return [
        SearchConfiguration(index=i, lr=float(c["lr"]), dropout=float(c["dropout"]))
        for i, c in enumerate(cfg["search"]["configurations"])
    ]


def _build_training(cfg: dict, config: SearchConfiguration, settings: TrainingSettings, device: torch.device, seed: int):
    seed_everything(seed)
    model = build_model(cfg, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.t_max, eta_min=settings.eta_min)
    scaler = torch.amp.GradScaler("cuda", enabled=settings.use_amp(device))
    return model, optimizer, scheduler, scaler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    grad_clip_norm: float,
    augment_cfg: dict | None = None,
    augment_seed: int | None = None,
) -> float:
    model.train()
    generator = None
    if augment_cfg is not None:
        generator = torch.Generator()
        generator.manual_seed(int(augment_seed))
    total, count = 0.0, 0
    for x, y, w, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        if augment_cfg is not None:
            x = spec_augment(x, augment_cfg, generator)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            logits = model(x).squeeze(1)
        loss = weighted_bce_loss(logits, y, w)
        if not torch.isfinite(loss):
            raise FloatingPointError("perdida no finita durante el entrenamiento")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * y.shape[0]
        count += int(y.shape[0])
    return total / max(count, 1)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> np.ndarray:
    model.eval()
    out = np.full(len(loader.dataset), np.nan, dtype=np.float64)
    for x, _y, _w, idx in loader:
        x = x.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(x).squeeze(1)
        out[idx.numpy()] = logits.float().cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("logits no finitos en inferencia")
    return out


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def bce_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """BCE media sin pesos (validation y test nunca se ponderan)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def evaluate_patients(
    segments: pd.DataFrame, probabilities: np.ndarray, negative_label_name: str, threshold: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Probabilidad por segmento -> media por grabacion -> media por paciente
    -> decision con ``threshold``. Devuelve metricas y las tres tablas."""
    scored = segments.assign(score=np.asarray(probabilities, dtype=np.float64))
    recordings = ev.aggregate_segment_to_recording(scored)
    patients = ev.aggregate_recording_to_patient(recordings)
    metrics = ev.compute_patient_metrics(
        patients["target_label"].to_numpy(), patients["score"].to_numpy(),
        negative_label_name=negative_label_name, threshold=threshold,
    )
    neg_key = f"recall_{negative_label_name.strip().lower()}"
    metrics["min_class_recall"] = float(min(metrics["recall_copd"], metrics[neg_key]))
    return metrics, scored, recordings, patients


def selection_key(metrics: dict) -> tuple[float, float, float, float]:
    """Mayor balanced accuracy, mayor macro-F1, mayor minimo recall entre
    clases y, por ultimo, menor perdida de validation. Un NaN nunca gana."""
    def finite(value, fallback):
        value = float(value)
        return value if math.isfinite(value) else fallback

    return (
        finite(metrics["balanced_accuracy"], -math.inf),
        finite(metrics["macro_f1"], -math.inf),
        finite(metrics["min_class_recall"], -math.inf),
        -finite(metrics["val_loss"], math.inf),
    )


def _release(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class ValidationRun:
    config: SearchConfiguration
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    best_metrics: dict
    history: pd.DataFrame
    normalization: NormalizationStats
    seconds: float


def train_with_validation(
    logmel: np.ndarray,
    train_seg: pd.DataFrame,
    val_seg: pd.DataFrame,
    cfg: dict,
    config: SearchConfiguration,
    settings: TrainingSettings,
    device: torch.device,
    negative_label_name: str,
    seed_parts: tuple,
    augment_cfg: dict | None = None,
    log=None,
) -> ValidationRun:
    """Entrena una configuracion en train, evalua validation por paciente al
    final de cada epoca y se detiene tras ``patience`` epocas sin mejorar la
    ``selection_key`` (nunca antes de ``min_epochs``). No guarda pesos: el
    fold reinicializa y reentrena sobre train + validation."""
    started = time.perf_counter()
    base = settings.seed_base
    use_amp = settings.use_amp(device)
    neg_key = f"recall_{negative_label_name.strip().lower()}"

    train_weights = dmod.compute_sample_weights(train_seg)
    stats = compute_normalization(
        logmel, train_seg["cache_row"],
        train_weights if settings.normalization_weighted else None, settings.std_floor,
    )
    model, optimizer, scheduler, scaler = _build_training(
        cfg, config, settings, device, derive_seed(base, *seed_parts, "init"),
    )
    train_ds = LogmelDataset(logmel, train_seg["cache_row"], train_seg["target_label"], train_weights, stats)
    train_loader = make_loader(
        train_ds, settings.batch_size, True, settings.num_workers,
        derive_seed(base, *seed_parts, "loader"), device,
        drop_last=_needs_drop_last(len(train_ds), settings.batch_size),
    )
    val_ds = LogmelDataset(logmel, val_seg["cache_row"], val_seg["target_label"], None, stats)
    val_loader = make_loader(
        val_ds, settings.eval_batch_size, False, settings.num_workers,
        derive_seed(base, *seed_parts, "val"), device,
    )
    val_labels = val_seg["target_label"].to_numpy(dtype=np.float64)

    rows, best_key, best_epoch, best_metrics = [], None, 0, {}
    stopped_early, epoch = False, 0
    for epoch in range(1, settings.max_epochs + 1):
        lr_epoch = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp, settings.grad_clip_norm,
            augment_cfg, derive_seed(base, *seed_parts, "augment", epoch),
        )
        scheduler.step()

        logits = predict_logits(model, val_loader, device, use_amp)
        metrics, *_ = evaluate_patients(val_seg, sigmoid(logits), negative_label_name, settings.threshold)
        metrics["val_loss"] = bce_from_logits(logits, val_labels)
        key = selection_key(metrics)
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_epoch, best_metrics = key, epoch, dict(metrics)

        rows.append({
            "config_index": config.index, "lr": config.lr, "dropout": config.dropout,
            "epoch": epoch, "lr_epoch": lr_epoch, "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_min_class_recall": metrics["min_class_recall"],
            "val_recall_copd": metrics["recall_copd"],
            f"val_{neg_key}": metrics[neg_key],
            "val_auroc": metrics["auroc"],
            "improved": improved,
        })
        if log is not None and (epoch == 1 or epoch % 10 == 0 or improved):
            log(
                f"config {config.index} (lr={config.lr:g}, dropout={config.dropout:g}) epoca {epoch}: "
                f"train_loss={train_loss:.4f} val_loss={metrics['val_loss']:.4f} "
                f"val_BA={metrics['balanced_accuracy']:.3f} mejor_epoca={best_epoch}"
            )
        if epoch >= settings.min_epochs and epoch - best_epoch >= settings.patience:
            stopped_early = True
            break

    del model, optimizer, scheduler, scaler, train_loader, val_loader
    _release(device)
    return ValidationRun(
        config=config, best_epoch=best_epoch, epochs_run=epoch, stopped_early=stopped_early,
        best_metrics=best_metrics, history=pd.DataFrame(rows), normalization=stats,
        seconds=time.perf_counter() - started,
    )


@dataclass
class FixedRun:
    model: CopdCNN
    normalization: NormalizationStats
    history: pd.DataFrame
    seconds: float


def train_fixed_epochs(
    logmel: np.ndarray,
    train_seg: pd.DataFrame,
    n_epochs: int,
    cfg: dict,
    config: SearchConfiguration,
    settings: TrainingSettings,
    device: torch.device,
    seed_parts: tuple,
    augment_cfg: dict | None = None,
    log=None,
) -> FixedRun:
    """Reentrena desde cero exactamente ``n_epochs`` epocas, sin validation:
    se usa para el reajuste train + validation de cada fold y para el modelo
    final. El scheduler usa el mismo ``T_max`` que la busqueda, de modo que el
    learning rate de cada epoca coincide con el de la epoca elegida."""
    if n_epochs < 1:
        raise ValueError("n_epochs debe ser >= 1")
    started = time.perf_counter()
    base = settings.seed_base
    use_amp = settings.use_amp(device)

    weights = dmod.compute_sample_weights(train_seg)
    stats = compute_normalization(
        logmel, train_seg["cache_row"],
        weights if settings.normalization_weighted else None, settings.std_floor,
    )
    model, optimizer, scheduler, scaler = _build_training(
        cfg, config, settings, device, derive_seed(base, *seed_parts, "init"),
    )
    dataset = LogmelDataset(logmel, train_seg["cache_row"], train_seg["target_label"], weights, stats)
    loader = make_loader(
        dataset, settings.batch_size, True, settings.num_workers,
        derive_seed(base, *seed_parts, "loader"), device,
        drop_last=_needs_drop_last(len(dataset), settings.batch_size),
    )

    rows = []
    for epoch in range(1, n_epochs + 1):
        lr_epoch = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(
            model, loader, optimizer, scaler, device, use_amp, settings.grad_clip_norm,
            augment_cfg, derive_seed(base, *seed_parts, "augment", epoch),
        )
        scheduler.step()
        rows.append({"epoch": epoch, "lr_epoch": lr_epoch, "train_loss": train_loss})
        if log is not None and (epoch == 1 or epoch % 10 == 0 or epoch == n_epochs):
            log(f"reajuste config {config.index} epoca {epoch}/{n_epochs}: train_loss={train_loss:.4f}")

    del optimizer, scheduler, scaler, loader
    return FixedRun(model=model, normalization=stats, history=pd.DataFrame(rows), seconds=time.perf_counter() - started)


# ---------------------------------------------------------------------------
# Un fold completo
# ---------------------------------------------------------------------------

@dataclass
class CnnFoldResult:
    fold: int
    selected: dict
    search_summary: pd.DataFrame
    search_history: pd.DataFrame
    refit_history: pd.DataFrame
    normalization_search: NormalizationStats
    normalization_refit: NormalizationStats
    model_state: dict
    segment_predictions: pd.DataFrame
    recording_predictions: pd.DataFrame
    patient_predictions: pd.DataFrame
    test_metrics: dict
    baseline_table: pd.DataFrame
    warnings: list[str]
    seconds: float


def run_cnn_fold(
    fold_id: int,
    condition: dmod.ConditionLogmel,
    train_patients: list[str],
    val_patients: list[str],
    test_patients: list[str],
    cfg: dict,
    configurations: list[SearchConfiguration],
    negative_label_name: str,
    device: torch.device,
    num_workers: int,
    hyperparameter_source: str = "search",
    log=None,
) -> CnnFoldResult:
    """Un fold:

    1. Por cada configuracion: entrenar en train, elegir su mejor epoca en
       validation (por paciente).
    2. Elegir la configuracion con la mejor ``selection_key`` en su mejor epoca
       (empate -> menor indice).
    3. Reinicializar y entrenar con train + validation exactamente esa epoca.
    4. Evaluar test una sola vez; linea base Dummy sobre el mismo test.

    Si ``configurations`` tiene una sola entrada (condicion con augmentation),
    el paso 2 no elige nada, pero la mejor epoca se sigue determinando en
    validation.
    """
    if not configurations:
        raise ValueError("no hay configuraciones que entrenar")
    started = time.perf_counter()
    spec, segments, logmel = condition.spec, condition.segments, condition.logmel
    train_seg = sp.filter_segments_by_patients(segments, train_patients)
    val_seg = sp.filter_segments_by_patients(segments, val_patients)
    test_seg = sp.filter_segments_by_patients(segments, test_patients)
    for name, part in (("train", train_seg), ("validation", val_seg), ("test", test_seg)):
        if part.empty:
            raise ValueError(f"fold {fold_id}: el conjunto {name} quedo vacio")
    groups = [set(train_seg["patient_uid"]), set(val_seg["patient_uid"]), set(test_seg["patient_uid"])]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError(f"fold {fold_id}: hay pacientes compartidos entre train, validation y test")

    settings = TrainingSettings.from_config(cfg, num_workers)
    augment_cfg = cfg["augmentation"] if spec.augment else None
    use_amp = settings.use_amp(device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        runs = [
            train_with_validation(
                logmel, train_seg, val_seg, cfg, config, settings, device, negative_label_name,
                seed_parts=(spec.dataset, fold_id, config.index, "search"), augment_cfg=augment_cfg, log=log,
            )
            for config in configurations
        ]
        best = max(runs, key=lambda r: (selection_key(r.best_metrics), -r.config.index))

        refit_seg = pd.concat([train_seg, val_seg], ignore_index=True)
        refit = train_fixed_epochs(
            logmel, refit_seg, best.best_epoch, cfg, best.config, settings, device,
            seed_parts=(spec.dataset, fold_id, best.config.index, "refit"), augment_cfg=augment_cfg, log=log,
        )

        test_ds = LogmelDataset(logmel, test_seg["cache_row"], test_seg["target_label"], None, refit.normalization)
        test_loader = make_loader(
            test_ds, settings.eval_batch_size, False, num_workers,
            derive_seed(settings.seed_base, spec.dataset, fold_id, "test"), device,
        )
        logits = predict_logits(refit.model, test_loader, device, use_amp)
        del test_loader
        metrics, scored, recordings, patients = evaluate_patients(
            test_seg, sigmoid(logits), negative_label_name, settings.threshold,
        )
        scored = scored.assign(logit=logits)

    model_state = {k: v.detach().cpu().clone() for k, v in refit.model.state_dict().items()}
    del refit.model
    _release(device)

    selected = {
        "fold": fold_id,
        "config_index": best.config.index,
        "lr": best.config.lr,
        "dropout": best.config.dropout,
        "best_epoch": best.best_epoch,
        "source": hyperparameter_source,
        "val_balanced_accuracy": best.best_metrics["balanced_accuracy"],
        "val_macro_f1": best.best_metrics["macro_f1"],
        "val_min_class_recall": best.best_metrics["min_class_recall"],
        "val_loss": best.best_metrics["val_loss"],
    }
    test_metrics = {
        **metrics, "fold": fold_id, "config_index": best.config.index,
        "lr": best.config.lr, "dropout": best.config.dropout, "epochs": best.best_epoch,
    }
    search_summary = pd.DataFrame([
        {
            "fold": fold_id, "config_index": r.config.index, "lr": r.config.lr, "dropout": r.config.dropout,
            "best_epoch": r.best_epoch, "epochs_run": r.epochs_run, "stopped_early": r.stopped_early,
            "val_balanced_accuracy": r.best_metrics["balanced_accuracy"],
            "val_macro_f1": r.best_metrics["macro_f1"],
            "val_min_class_recall": r.best_metrics["min_class_recall"],
            "val_loss": r.best_metrics["val_loss"],
            "seconds": r.seconds, "selected": r.config.index == best.config.index,
        }
        for r in runs
    ])
    search_history = pd.concat([r.history.assign(fold=fold_id) for r in runs], ignore_index=True)

    return CnnFoldResult(
        fold=fold_id,
        selected=selected,
        search_summary=search_summary,
        search_history=search_history,
        refit_history=refit.history.assign(fold=fold_id, config_index=best.config.index),
        normalization_search=runs[0].normalization,
        normalization_refit=refit.normalization,
        model_state=model_state,
        segment_predictions=scored,
        recording_predictions=recordings,
        patient_predictions=patients,
        test_metrics=test_metrics,
        baseline_table=compute_dummy_baseline(refit_seg, test_seg, negative_label_name, fold_id),
        warnings=sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# Modelo final
# ---------------------------------------------------------------------------

def median_epochs(values) -> int:
    """Mediana redondeada al entero mas cercano, minimo 1."""
    return max(1, int(math.floor(float(np.median(np.asarray(values, dtype=np.float64))) + 0.5)))


def select_final_configuration(search_summaries: pd.DataFrame) -> tuple[SearchConfiguration, int, pd.DataFrame]:
    """Promedia entre folds las metricas de validation de cada configuracion
    (en su mejor epoca), elige con el mismo orden que ``selection_key`` y
    devuelve tambien la mediana de las mejores epocas de esa configuracion.
    Nunca mira metricas de test."""
    grouped = search_summaries.groupby("config_index", as_index=False).agg(
        lr=("lr", "first"),
        dropout=("dropout", "first"),
        val_balanced_accuracy=("val_balanced_accuracy", "mean"),
        val_macro_f1=("val_macro_f1", "mean"),
        val_min_class_recall=("val_min_class_recall", "mean"),
        val_loss=("val_loss", "mean"),
        median_best_epoch=("best_epoch", "median"),
        n_folds=("fold", "nunique"),
    )
    ranked = grouped.assign(neg_val_loss=-grouped["val_loss"]).sort_values(
        ["val_balanced_accuracy", "val_macro_f1", "val_min_class_recall", "neg_val_loss", "config_index"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    top = ranked.iloc[0]
    config = SearchConfiguration(index=int(top["config_index"]), lr=float(top["lr"]), dropout=float(top["dropout"]))
    epochs = median_epochs(search_summaries.loc[search_summaries["config_index"] == config.index, "best_epoch"])
    return config, epochs, grouped


def train_final_model(
    condition: dmod.ConditionLogmel,
    cfg: dict,
    config: SearchConfiguration,
    n_epochs: int,
    device: torch.device,
    num_workers: int,
    log=None,
) -> FixedRun:
    """Todos los pacientes elegibles de la condicion, ``n_epochs`` epocas."""
    settings = TrainingSettings.from_config(cfg, num_workers)
    augment_cfg = cfg["augmentation"] if condition.spec.augment else None
    return train_fixed_epochs(
        condition.logmel, condition.segments, n_epochs, cfg, config, settings, device,
        seed_parts=(condition.spec.dataset, "final", config.index), augment_cfg=augment_cfg, log=log,
    )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_payload(
    state_dict: dict,
    normalization: NormalizationStats,
    cfg: dict,
    spec: dmod.ConditionSpec,
    negative_label_name: str,
    hyperparameters: dict,
    extra: dict | None = None,
) -> dict:
    """Todo lo necesario para reconstruir el modelo e inferir sin el repo de
    configuracion: arquitectura, pesos, normalizacion, Log-Mel, clases y
    umbral. Solo contiene tensores y tipos nativos, de modo que se carga con
    ``torch.load(..., weights_only=True)``."""
    architecture = architecture_description(cfg)
    architecture["dropout"] = float(hyperparameters["dropout"])
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture,
        "state_dict": state_dict,
        "normalization": normalization.to_tensors(),
        "input_shape": architecture["input_shape"],
        "acoustic_config": dict(cfg["acoustic"]),
        "logmel_config": dict(cfg["logmel"]),
        "class_names": [negative_label_name, "COPD"],
        "label_mapping": {negative_label_name: 0, "COPD": 1},
        "output": "logit; sigmoid(logit) = P(COPD) por segmento",
        "aggregation": "media por grabacion, luego media por paciente",
        "decision_threshold": float(cfg["evaluation"]["decision_threshold"]),
        "dataset": spec.dataset,
        "condition": spec.condition,
        "branch": spec.branch,
        "augment": bool(spec.augment),
        "hyperparameters": {
            "config_index": int(hyperparameters["config_index"]),
            "lr": float(hyperparameters["lr"]),
            "dropout": float(hyperparameters["dropout"]),
            "epochs": int(hyperparameters["epochs"]),
        },
        **(extra or {}),
    }


def save_checkpoint(path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def model_from_checkpoint(payload: dict) -> nn.Module:
    """Reconstruye la CNN o la CRNN a partir de ``payload["architecture"]``.

    Una clase desconocida se rechaza de forma explicita: asumir que es una CNN
    produciria un error confuso al cargar los pesos o, peor, una red distinta.
    """
    architecture = payload["architecture"]
    from . import crnn as crnn_arch

    class_name = architecture.get("class")
    if class_name == crnn_arch.ARCHITECTURE_NAME:
        model = crnn_arch.model_from_architecture(architecture)
    elif class_name == ARCHITECTURE_NAME:
        model = CopdCNN(
            in_channels=int(architecture["in_channels"]),
            channels=tuple(int(c) for c in architecture["channels"]),
            hidden_units=int(architecture["hidden_units"]),
            dropout=float(architecture["dropout"]),
        )
    else:
        raise ValueError(
            f"clase de arquitectura desconocida en el checkpoint: {class_name!r} "
            f"(se admiten {ARCHITECTURE_NAME!r} y {crnn_arch.ARCHITECTURE_NAME!r})"
        )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
