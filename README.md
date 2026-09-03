# Detección de enfermedades pulmonares mediante modelos de Inteligencia Artificial a partir de grabaciones de audio respiratorio

Proyecto de Fin de Carrera · Pontificia Universidad Católica del Perú
Facultad de Ciencias e Ingeniería

**Autor:** Sivincha Bailón, Josué Manuel

---

## Descripción

El proyecto desarrolla modelos de Inteligencia Artificial capaces de detectar enfermedades
pulmonares a partir de sonidos respiratorios captados por auscultación. Uno de sus objetivos
centrales es analizar la influencia del dispositivo de captura sobre la detección, y
seleccionar un modelo robusto ante la heterogeneidad instrumental.

El repositorio reúne los dos corpus de audio empleados, su metadata unificada y el pipeline
de preprocesamiento que garantiza la consistencia de las señales antes de la extracción de
características.

---

## Los datos

| | ICBHI 2017 | Fraiwan (Mendeley) | Total |
|---|---:|---:|---:|
| Archivos de audio | 920 | 336 | **1256** |
| Grabaciones independientes | 920 | 112 | **1032** |
| Pacientes | 126 | 112 | **238** |
| Diagnósticos distintos | 8 | 11 | 15 |
| Dispositivos de captura | 4 | 1 | 4 |
| Zonas de auscultación | 7 | 10 | — |
| Frecuencia de muestreo | 44.1 / 10 / 4 kHz | 4 kHz | — |
| Duración total | 5.49 h | 1.62 h | 7.11 h |

ICBHI aporta además **6898 ciclos respiratorios anotados** con marcas temporales de
crepitantes y sibilancias.

En Fraiwan los 336 archivos corresponden a 112 grabaciones almacenadas en tres modos de
filtrado del mismo instante acústico, por lo que el número de unidades independientes es
112 y no 336.

Los detalles de cada corpus, el esquema de la metadata y los sesgos conocidos están
documentados en [`Audios_Respiratorios/README.md`](Audios_Respiratorios/README.md).

---

## Estructura del repositorio

```
Pulmonary-Disease/
├── Audios_Respiratorios/          Datos originales y metadata unificada
│   ├── Dataset1-ICBHI/
│   │   ├── audio/                 920 .wav organizados por dispositivo
│   │   ├── annotations/           920 .txt de ciclos respiratorios
│   │   ├── metadata/              CSV de metadata, ciclos y resumen
│   │   └── source_archive/        Enlace al .zip original
│   └── Dataset2_Mendeley/
│       ├── audio/                 336 .wav por modo de filtrado
│       ├── metadata/              CSV de metadata y anotación clínica
│       └── source_archive/        Enlace al .zip original
│
└── preprocessing/                 Pipeline de preprocesamiento (Resultado 2)
    ├── reports/                   Informes de verificación y figuras
    └── data/                      Audio procesado y segmentos (no versionado)
```

Los archivos `.zip` originales de ambos datasets no se versionan por superar el límite de
tamaño de GitHub. Se conservan en Google Drive; el enlace figura en los archivos
`source_archive/Dataset1.txt` y `source_archive/Dataset2.txt`.

---

## Reproducción

**Requisitos:** Python 3.11 o superior, con `numpy`, `scipy`, `pandas` y `soundfile`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy pandas soundfile
```

Los audios procesados no se versionan por ser derivables: se regeneran ejecutando el
pipeline sobre los datos originales. El procedimiento completo está documentado en
[`preprocessing/README.md`](preprocessing/README.md).

---

## Estado del proyecto

| Resultado esperado | Estado |
|---|---|
| 1 · Metadata unificada de ambos corpus | Completo |
| 2 · Pipeline de preprocesamiento validado | Documentado y validado; implementación en curso |
| 3 · Entrenamiento y comparación de modelos | Pendiente |
| 4 · Benchmarking entre dispositivos de captura | Pendiente |

---

## Fuentes de los datos

**ICBHI 2017 Respiratory Sound Database** — Rocha et al. Base de datos del *International
Conference on Biomedical and Health Informatics Challenge*, recolectada en Portugal y
Grecia.

**Respiratory sound dataset** — Fraiwan et al. Grabaciones obtenidas en Jordania mediante
estetoscopio electrónico 3M Littmann 3200. Publicado en Mendeley Data.

Ambos conjuntos son de acceso público y se emplean conforme a sus términos de uso
originales. Las citas formales completas deben consultarse en las publicaciones de origen.
