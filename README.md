# SATDM — Inspector de Dispositivos Médicos (Visión + Audio)

Proyecto final del módulo **TAE-IA · Módulo 6 — Aplicaciones de Deep Learning en Tiempo Real**
(Cinvestav Guadalajara). Extiende el proyecto de investigación **SATDM** (Sistema Auditable de
Trazabilidad de Dispositivos Médicos) con un pipeline multimodal de inspección visual y acústica
servido en una interfaz Gradio.

> App Gradio que, a partir de una foto/video de un dispositivo médico y una nota de voz opcional
> del técnico, detecta y localiza el dispositivo, clasifica su tipo y estado físico, genera una
> descripción en lenguaje natural, analiza el evento acústico del entorno, transcribe la
> observación verbal del operador, y registra cada inspección en una bitácora auditable con hash
> encadenado.

## Tabla de contenido

- [Arquitectura](#arquitectura)
- [Modelos utilizados](#modelos-utilizados)
- [Requisitos](#requisitos)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Configuración](#configuración)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Bitácora de trazabilidad](#bitácora-de-trazabilidad)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Ética y guardrails](#ética-y-guardrails)
- [Créditos](#créditos)

## Arquitectura

Pipeline mixto (secuencial con una rama condicional de rechazo temprano):

```
Imagen/Video ──► YOLOv8 (detección, siempre corre)
             ──► BLIP   (captioning, siempre corre)
             ──► Gatekeeper BiomedCLIP zero-shot (umbral 0.55)
                     │
              ┌──────┴───────┐
              │               │
           no pasa          pasa
              │               │
              ▼               ▼
          Rechazo      BiomedCLIP (tipo + estado)
              │               │
              │        Audio ─┤ CLAP (evento acústico)
              │        opcional─ Whisper (transcripción ASR)
              │               │
              └───────┬───────┘
                       ▼
          Reporte de texto integrado + audio TTS opcional (XTTS, desactivado por defecto)
                       │
                       ▼
       Bitácora CSV + evidencia JSON — hash SHA-256 encadenado (se escribe SIEMPRE)
```

Puntos clave:
- **YOLO y BLIP corren siempre** sobre la imagen, pase o no el gatekeeper.
- **El gatekeeper solo evalúa la imagen** (BiomedCLIP zero-shot contra un set de etiquetas de
  dominio médico). Si no pasa, **CLAP y Whisper no se ejecutan** — la inspección se rechaza y se
  audita igual, sin analizar el audio.
- **Toda inspección se registra en la bitácora**, sea aprobada o rechazada.

## Modelos utilizados

| Modelo | Tarea | Notas |
|---|---|---|
| YOLOv8 (pesos propios, fallback `yolov8n.pt`) | Detección/localización | Entrenado sobre dataset propio de dispositivos médicos (Roboflow); si no encuentra los pesos custom en Drive, cae automáticamente a YOLOv8n genérico (COCO) sin detener la app |
| BiomedCLIP (`open_clip`, zero-shot) | Gatekeeper de dominio + clasificación de tipo/estado | Doble uso: filtra si la imagen es médica y, si pasa, clasifica tipo de dispositivo y estado físico |
| BLIP (`blip-image-captioning-large`) | Captioning | Descripción en lenguaje natural que complementa la clasificación estructurada |
| CLAP (`laion/clap-htsat-unfused`, zero-shot) | Clasificación de eventos acústicos | Alarma, motor, voz, ambiente, silencio — solo corre si el gatekeeper de imagen aprobó |
| Whisper (`openai/whisper-small`) | Transcripción de voz (ASR) | Transcribe la nota de voz del técnico; también alimenta un *override* por palabras clave de tipo/estado |
| XTTS v2 (opcional, `ENABLE_TTS`) | Síntesis de voz del reporte | **Desactivado por defecto** — el requisito de audio ya se cumple con CLAP + Whisper en la entrada |

Justificación completa (tamaño, por qué ese modelo y no otro) en `docs/M6_Reporte_SATDM.docx` / `.pdf`.

## Requisitos

- Python 3.10+ (probado en Google Colab)
- GPU opcional (el pipeline corre en CPU con `DEVICE="cpu"`, solo más lento)
- Cuenta de Google Drive si se quiere usar los pesos YOLO propios (`pesos_medicos_yolov8.pt`)
- Ver `requirements.txt` para el listado exacto de paquetes

## Instalación y ejecución

### En Google Colab (recomendado, forma en que se probó)

1. Abrir `app_inspector_SATDM_Produccion_FINAL.ipynb` en Colab.
2. Montar Google Drive y colocar `pesos_medicos_yolov8.pt` en la carpeta configurada
   (`DRIVE_ROOT` en la sección de configuración). Si no se coloca, la app usa `yolov8n.pt`
   genérico como respaldo, sin detenerse.
3. Ejecutar las celdas en orden, de arriba hacia abajo (Setup → Configuración → Modelos →
   Taxonomías → Utilitarias → Inferencia → Inspección imagen/video → Interfaz).
4. Ejecutar la última celda (`app.launch(share=True, debug=True)`) y abrir la URL pública que
   se imprime en consola.
5. Para detener, interrumpir la celda (botón de stop) — cierra el túnel de Gradio de forma segura.

### Local

```bash
git clone <url-del-repo>
cd satdm-inspector
pip install -r requirements.txt
jupyter notebook app_inspector_SATDM_Produccion_FINAL.ipynb
```

Ejecutar las celdas en el mismo orden que en Colab. La primera celda descarga los pesos de
Hugging Face para BiomedCLIP, BLIP, CLAP y Whisper (requiere conexión a internet la primera vez).

## Configuración

Parámetros clave definidos en la sección de configuración del notebook:

| Parámetro | Valor | Qué controla |
|---|---|---|
| `MEDICAL_THRESHOLD` | `0.55` | Confianza mínima para aceptar una clasificación de tipo/estado sin marcarla para revisión manual, y umbral del gatekeeper |
| `AMBIGUITY_ENTROPY` | `0.85` | Entropía normalizada por encima de la cual una clasificación se marca como ambigua (`review_required`) |
| `MIN_SIDE` / `MAX_SIDE` | `64` / `1280` px | Rango de tamaño de imagen aceptado antes de rechazar o reescalar |
| `YOLO_CONF` | `0.25` | Confianza mínima de detección de YOLO |
| `VIDEO_FRAME_STRIDE` | `15` | Cada cuántos frames se re-clasifica un track en el modo video |
| `CLAP_SR` | `48000` Hz | Sample rate objetivo para CLAP |
| `AUDIO_MIN_SECONDS` / `AUDIO_MAX_SECONDS` | `0.5` / `60` s | Duración aceptada de un clip de audio |
| `AUDIO_CONF_FLOOR` | `0.55` | Confianza mínima de CLAP; por debajo se reporta `no_concluyente` en vez de forzar una etiqueta |
| `VOICE_CONF_OVERRIDE` | `0.85` | Confianza asignada cuando el tipo/estado se decide por voz del operador en vez de por visión (calibrada para no anular `review_required` artificialmente) |
| `ENABLE_TTS` | `False` | Activa/desactiva la síntesis de voz del reporte con XTTS |
| `SEED` | `42` | Semilla fija para `random`, `numpy` y `torch` (reproducibilidad) |

## Estructura del repositorio

```
.
├── app_inspector_SATDM_Produccion_FINAL.ipynb   # Notebook principal (modelos + Gradio)
├── requirements.txt                              # Dependencias exactas
├── docs/
│   └── M6_Reporte_SATDM.docx                     # Reporte técnico completo del proyecto
├── inspecciones_log.csv                          # Bitácora generada en tiempo de ejecución (no versionar con datos reales)
├── evidence/                                     # JSON de evidencia por inspección (no versionar con datos reales)
└── README.md
```

## Bitácora de trazabilidad

Cada inspección —aprobada o rechazada por el gatekeeper— se escribe en `inspecciones_log.csv`
con, entre otros, estos campos:

- `inspection_id`, `timestamp_utc`, `device_id`
- `tipo_detectado`, `tipo_confianza`, `tipo_entropia`
- `estado_detectado`, `estado_confianza`, `estado_entropia`
- `gatekeeper_passed`, `gatekeeper_score`, `review_required`
- `evidence_sha256`, hash **encadenado** al registro anterior

El encadenamiento de hashes hace que alterar retroactivamente un registro rompa la cadena y sea
detectable — es la base de la propiedad "auditable" del sistema.

## Limitaciones conocidas

- Corre en zero-shot (BiomedCLIP/CLAP) sin validación cuantitativa propia todavía: no hay un
  conjunto de prueba etiquetado que mida con números la matriz de confusión entre estados
  visualmente similares (p. ej. "sucio" vs. "corrosión").
- El *override* por voz depende de coincidencia de subcadenas exactas contra un diccionario de
  palabras clave; un error de transcripción de Whisper puede romper el match sin avisar por qué.
- El seguimiento de objetos en video (ByteTrack) puede reasignar el ID de un track tras una
  oclusión, perdiendo el historial de clasificación acumulado de ese objeto.
- No hay autenticación de operador ni verificación de que el dispositivo fotografiado corresponda
  a un número de serie específico del inventario — ver sección de Ética del reporte técnico.

Detalle completo, con hipótesis de causa raíz y mitigación propuesta, en el reporte técnico
(`docs/M6_Reporte_SATDM.docx`, sección "Casos de falla").

## Ética y guardrails

- Bitácora *append-only* con hash SHA-256 encadenado — alterar un registro pasado rompe la cadena.
- Toda inspección se audita, sea aprobada o rechazada por el gatekeeper.
- `review_required` se activa automáticamente ante baja confianza o alta ambigüedad, forzando
  revisión humana.
- Síntesis de voz (XTTS) **desactivada por defecto** para evitar el escenario de clonación de voz
  sin consentimiento.
- Problema abierto: el sistema no autentica al operador ni liga criptográficamente una inspección
  al número de serie real del dispositivo — ver reporte técnico para el análisis completo.

## Créditos

Proyecto elaborado por **Juan Carlos Aquino Hernández** — División de Electromecánica Industrial,
Universidad Tecnológica de Nayarit (UTNay) — como parte del curso TAE-IA (Cinvestav Guadalajara)
y del proyecto de investigación SATDM.
