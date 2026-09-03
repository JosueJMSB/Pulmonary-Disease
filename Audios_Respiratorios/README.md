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

## Sesgos conocidos

Documentados aquí para que ningún análisis posterior los pase por alto.

### 1 · El dispositivo predice el diagnóstico en ICBHI

| Dispositivo | Audios | % que es COPD |
|---|---:|---:|
| AKG C417L | 646 | **100 %** |
| Littmann 3200 | 60 | **100 %** |
| Littmann C2SE | 87 | 58.6 % |
| Meditron | 127 | 28.3 % |

706 de los 793 audios de COPD provienen de dos micrófonos que no registraron ninguna otra
patología. Un modelo puede aprender a identificar el hardware en lugar de la enfermedad.
Solo 4 de los 126 pacientes fueron grabados con más de un dispositivo.

La frecuencia de muestreo y la profundidad de bits también quedan determinadas por el
dispositivo, lo que añade capas al mismo confound.

### 2 · El desbalance depende de la unidad de conteo

La EPOC representa el **86.2 %** de los audios de ICBHI pero solo el **50.8 %** de los
pacientes: a cada paciente con EPOC se le registraron 12.4 audios en promedio, frente a 1.3
en los sanos. Las cifras a nivel de paciente son las que deben reportarse.

### 3 · Fuga de información por duración en ICBHI

Todas las clases salvo la EPOC están fijadas en 20 segundos. De las **95 grabaciones que se
salen de ese valor, las 95 son EPOC**. Un clasificador que solo mire la duración del archivo
acierta esos 95 casos sin procesar audio. Las entradas deben llevarse a longitud fija y la
duración nunca debe usarse como característica.

### 4 · La zona lleva información diagnóstica en Fraiwan

De los 72 audios de la zona `PRL` (posterior derecha inferior), 42 corresponden a
insuficiencia cardíaca, EPOC o su combinación, y solo 6 a personas sanas. Tiene sentido
clínico —es donde se auscultan los crepitantes basales— pero convierte la posición del
estetoscopio en un segundo confound. Como cada paciente aporta una sola zona, el efecto no
puede controlarse dentro del paciente.

### 5 · El diagnóstico es una etiqueta débil

El diagnóstico describe al paciente, no al fragmento de audio. En ICBHI, el **47.4 % de los
ciclos de pacientes con EPOC no presenta ningún hallazgo anotado**, y 215 de sus 793
grabaciones no contienen un solo evento en toda su duración. Cualquier segmento hereda la
etiqueta del paciente sin garantía de contener evidencia acústica de la enfermedad.

### 6 · La edad actúa como variable de confusión

La duración mediana del ciclo respiratorio varía de 1.43 s en bronquiolitis a 3.06 s en
bronquiectasia, lo que corresponde a frecuencias respiratorias de 42 y 20 por minuto
respectivamente. No es una propiedad acústica de esas enfermedades: la bronquiolitis y la
LRTI son patologías pediátricas y los lactantes respiran mucho más rápido. El ritmo
respiratorio distingue clases sin necesidad de escuchar el pulmón.

### 7 · Los sistemas de zonas no son compatibles entre corpus

ICBHI incluye plano lateral y tráquea, que Fraiwan no tiene; Fraiwan subdivide en superior,
medio e inferior, lo que ICBHI no hace. Al unificar solo sobreviven cuatro zonas comunes
—`AL`, `AR`, `PL`, `PR`— descartando 319 audios de ICBHI (el 34.7 %) y colapsando los tres
niveles verticales de Fraiwan.

---

## Archivos originales

Los `.zip` de ambos datasets no se versionan por superar el límite de tamaño de GitHub. El
de ICBHI contiene además los mismos audios duplicados. Se conservan en Google Drive; el
enlace está en `Dataset1-ICBHI/source_archive/Dataset1.txt` y
`Dataset2_Mendeley/source_archive/Dataset2.txt`.
