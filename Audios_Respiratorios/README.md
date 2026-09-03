# Diccionario de datos

Documentación de los dos corpus de audio respiratorio del proyecto: origen, organización,
esquema de la metadata y sesgos conocidos.

---

## Dataset 1 · ICBHI 2017 Respiratory Sound Database

Base de datos del *International Conference on Biomedical and Health Informatics Challenge*,
recolectada en Portugal y Grecia. Es el conjunto de referencia en clasificación de sonidos
respiratorios.

| | |
|---|---|
| Archivos de audio | 920 |
| Pacientes | 126 |
| Diagnósticos | 8 |
| Dispositivos | 4 |
| Zonas de auscultación | 7 |
| Frecuencia de muestreo | 44 100 Hz (824), 4000 Hz (90), 10 000 Hz (6) |
| Profundidad de bits | 24 bits (792), 16 bits (128) |
| Canales | Mono |
| Duración | 5.49 h · mediana 20.0 s · rango 7.9–86.2 s |
| Anotaciones | 6898 ciclos respiratorios con marcas temporales |

### Distribución por diagnóstico

| Diagnóstico | Audios | Pacientes |
|---|---:|---:|
| COPD | 793 | 64 |
| Pneumonia | 37 | 6 |
| Healthy | 35 | 26 |
| URTI | 23 | 14 |
| Bronchiectasis | 16 | 7 |
| Bronchiolitis | 13 | 6 |
| LRTI | 2 | 2 |
| Asthma | 1 | 1 |

### Organización

Los audios se agrupan por dispositivo de captura:

```
audio/
├── 1_3M Littmann 3200 Electronic/                    60 archivos
├── 2_3M Littmann Classic II SE/                      87 archivos
├── 3_AKG C417L/                                     646 archivos
└── 4_WelchAllyn Meditron Master Elite Electronic/   127 archivos
```

`annotations/` replica esa misma estructura con un `.txt` por grabación.

### Nomenclatura de archivos

```
101_URTI_AL_Meditron_1b1.wav
 │    │    │      │      └── índice de grabación
 │    │    │      └───────── dispositivo
 │    │    └──────────────── zona de auscultación
 │    └───────────────────── diagnóstico
 └────────────────────────── identificador de paciente
```

Difiere de la nomenclatura original de ICBHI, que era
`{paciente}_{grabación}_{zona}_{modo}_{dispositivo}`. La correspondencia entre ambas fue
verificada por dos vías independientes —parseo de campos y emparejamiento por hash MD5— con
coincidencia total en las 920 grabaciones.

### Zonas de auscultación

| Código | Localización |
|---|---|
| `TC` | Tráquea |
| `AL` / `AR` | Anterior izquierda / derecha |
| `PL` / `PR` | Posterior izquierda / derecha |
| `LL` / `LR` | Lateral izquierda / derecha |

### Formato de las anotaciones

Cada `.txt` contiene una línea por ciclo respiratorio, con cuatro columnas separadas por
tabulador y sin encabezado:

```
0.364    3.250    0    1
  │        │      │    └── sibilancias (0/1)
  │        │      └─────── crepitantes (0/1)
  │        └────────────── fin del ciclo en segundos
  └─────────────────────── inicio del ciclo en segundos
```

Distribución de los 6898 ciclos: 3642 normales, 1864 con crepitantes, 886 con sibilancias
y 506 con ambos. Estas cifras reproducen las publicadas para ICBHI 2017.

---

## Dataset 2 · Fraiwan (Mendeley Data)

Grabaciones obtenidas en Jordania mediante estetoscopio electrónico 3M Littmann 3200,
publicadas en Mendeley Data.

| | |
|---|---|
| Archivos de audio | 336 |
| **Grabaciones independientes** | **112** |
| Pacientes | 112 |
| Diagnósticos | 11 |
| Dispositivos | 1 (3M Littmann 3200) |
| Zonas de auscultación | 10 |
| Frecuencia de muestreo | 4000 Hz |
| Profundidad de bits | 16 bits |
| Canales | Mono |
| Duración | 1.62 h · mediana 16.0 s · rango 5.0–30.0 s |
| Anotaciones de ciclo | No disponibles |

### Los tres modos de filtrado

Cada grabación se almacena tres veces, correspondientes a los modos de filtrado del
estetoscopio aplicados **al mismo instante acústico**:

```
audio/3M_Littmann_3200/
├── 1_Bell_20-200Hz/          112 archivos · sonidos graves
├── 2_Diaphragm_100-500Hz/    112 archivos · ruidos respiratorios
└── 3_Extended_50-500Hz/      112 archivos · banda ampliada
```

Los 336 archivos **no son 336 observaciones independientes**. El corpus contiene 112
instantes acústicos, y los tres modos de un mismo paciente deben permanecer siempre en la
misma partición de entrenamiento o prueba.

### Distribución por diagnóstico

| Diagnóstico | Audios | Pacientes |
|---|---:|---:|
| Normal | 105 | 35 |
| Asthma | 96 | 32 |
| HeartFailure | 54 | 18 |
| COPD | 27 | 9 |
| Pneumonia | 15 | 5 |
| LungFibrosis | 12 | 4 |
| Bronchitis | 9 | 3 |
| HeartFailure-COPD | 6 | 2 |
| PleuralEffusion | 6 | 2 |
| HeartFailure-LungFibrosis | 3 | 1 |
| Asthma-LungFibrosis | 3 | 1 |

Incluye patología cardíaca y cuatro diagnósticos combinados que ICBHI no contempla.

### Zonas de auscultación

Los códigos siguen el patrón `{A,P}{L,R}{U,M,L}`: anterior o posterior, izquierda o derecha,
superior, medio o inferior. Cada paciente aporta **una sola zona**.

---

## Esquema de la metadata

`icbhi_audio_metadata.csv` y `fraiwan_audio_metadata.csv` comparten las mismas 21 columnas,
lo que permite concatenarlos directamente.

| Columna | Descripción |
|---|---|
| `dataset` | `ICBHI` o `FRAIWAN` |
| `audio_id` | Identificador único del archivo |
| `filename`, `audio_path` | Nombre y ruta relativa |
| `patient_id`, `patient_uid` | Identificador de paciente, con prefijo del corpus |
| `diagnosis` | Diagnóstico clínico |
| `zone` | Zona de auscultación |
| `device` | Dispositivo de captura |
| `recording_id` | Índice de grabación (solo ICBHI) |
| `acquisition_mode` | Secuencial (`sc`) o multicanal (`mc`) |
| `filter`, `filter_low_hz`, `filter_high_hz` | Modo de filtrado (solo Fraiwan) |
| `sound_type` | Anotación clínica del sonido (solo Fraiwan) |
| `age`, `gender` | Datos demográficos |
| `duration_seconds`, `sample_rate_hz`, `channels`, `bit_depth` | Propiedades técnicas |

### Archivos derivados de ICBHI

| Archivo | Contenido |
|---|---|
| `icbhi_respiratory_cycles.csv` | 6898 filas, una por ciclo: tiempos, banderas de evento y etiqueta |
| `icbhi_cycle_summary.csv` | 920 filas, agregados por grabación: conteos, proporciones y cobertura |
| `patient_diagnosis.csv` | Diagnóstico por paciente, tal como viene del corpus original |
| `demographic_info.txt` | Edad, sexo, peso y talla por paciente |

---

