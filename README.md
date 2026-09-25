# SATDM — Medical Device Inspector (Vision + Audio)

[![Python 3.10+](https://shields.io)](https://python.org)
[![Platform-Colab](https://shields.io)](https://google.com)
[![License-Public_Domain](https://shields.io)](https://unlicense.org)

Final project for the module **TAE-IA · Module 6 — Real-Time Deep Learning Applications** (Cinvestav Guadalajara). It extends the **SATDM** (Medical Device Auditable Traceability System) research project with a multimodal visual and acoustic inspection pipeline served through a Gradio interface.

> A Gradio app that, based on a photo/video of a medical device and an optional technician voice note, detects and localizes the device, classifies its type and physical status, generates a natural language description, analyzes environmental acoustic events, transcribes the operator's verbal observation, and logs every inspection in an auditable ledger with a chained hash.

## Table of Contents

- [Architecture](#architecture)
- [Models Used](#models-used)
- [Requirements](#requirements)
- [Installation and Execution](#installation-and-execution)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Traceability Log Ledger](#traceability-log-ledger)
- [Known Limitations](#known-limitations)
- [Ethics and Guardrails](#ethics-and-guardrails)
- [Credits](#credits)

## Architecture

Mixed pipeline (sequential with a conditional early rejection branch):


### Key Principles:
- **YOLO and BLIP always run** on the image, regardless of whether it passes the gatekeeper or not.
- **The gatekeeper only evaluates the image** (BiomedCLIP zero-shot against a set of medical domain labels). If it does not pass, **CLAP and Whisper are bypassed** — the inspection is rejected and logged anyway, without processing the audio.
- **Every inspection is recorded in the ledger**, whether it is approved or rejected.

## Models Used

| Model | Task | Operational Notes |
|---|---|---|
| **YOLOv8** <br>*(Custom Weights / Fallback `yolov8n.pt`)* | Object Detection & Localization | Trained on a proprietary dataset of medical devices (Roboflow); if the custom weights are not found in Drive, it automatically falls back to generic COCO-trained YOLOv8n without breaking the app context. |
| **BiomedCLIP** <br>*(`open_clip`, zero-shot)* | Domain Gatekeeper + Type/Status Classification | Dual-purpose implementation: primarily acts as the domain gatekeeper filter and, upon clearance, classifies categorical device type and physical status. |
| **BLIP** <br>*(`blip-image-captioning-large`)* | Contextual Captioning | Generates unstructured, natural language descriptions that complement the structured classification tokens. |
| **CLAP** <br>*(`laion/clap-htsat-unfused`, zero-shot)* | Acoustic Event Classification | Alarm, engine, voice, environment, silence — runs strictly if the visual gatekeeper approves the device domain. |
| **Whisper** <br>*(`openai/whisper-small`)* | Automated Speech Recognition (ASR) | Transcribes verbal voice annotations from technicians and feeds an experimental dictionary-based keyword override for types and conditions. |
| **XTTS v2** <br>*(Optional, `ENABLE_TTS`)* | Speech Synthesis (TTS) | **Disabled by default** — the multi-modal audio input requirement is already met with CLAP + Whisper at the input stage. |

> 📑 *For detailed sizing metrics, benchmarking, and cross-model design trade-offs, please consult the complete technical report file located in `docs/M6_Reporte_SATDM.docx` or `docs/M6_Reporte_SATDM.pdf`.*

## Requirements

- Python 3.10+ (Fully validated on Google Colab ecosystems).
- GPU acceleration is optional (the pipeline falls back cleanly to CPU execution with `DEVICE="cpu"`, yielding higher latency).
- A Google Drive account is required if you intend to map the custom fine-tuned weights file (`pesos_medicos_yolov8.pt`).
- See `requirements.txt` for the frozen environment dependency manifest.

## Installation and Execution

### Option A: Google Colab Deployment (Recommended Reference Platform)

1. Open the `app_inspector_SATDM_Produccion_FINAL.ipynb` notebook inside your Colab environment.
2. Mount your Google Drive storage path and place your custom weights file (`pesos_medicos_yolov8.pt`) inside the configured directory (`DRIVE_ROOT` under the settings module). If unmapped, the app defaults to the standard generic `yolov8n.pt` asset automatically without stopping.
3. Step through the workspace cells sequentially from top to bottom: **Setup** ➔ **Configuration** ➔ **Models** ➔ **Taxonomies** ➔ **Utilities** ➔ **Inference** ➔ **Image/Video Inspection** ➔ **Interface**.
4. Trigger the final UI deployment block cell (`app.launch(share=True, debug=True)`) and click on the generated public URL printed in the console output stack.
5. To shut down the interface instance, interrupt the running execution cell (Stop button) to close the secure Gradio network tunnel clean.

### Option B: Local Environment Deployment

```bash
# Clone the repository asset tree
git clone <your-repository-url>
cd satdm-inspector

# Install dependencies
pip install -r requirements.txt

# Launch the notebook server
jupyter notebook app_inspector_SATDM_Produccion_FINAL.ipynb
```

Step through the execution blocks in the exact same order specified in the Colab workflow. The initial model activation routine will download weights from Hugging Face servers for BiomedCLIP, BLIP, CLAP, and Whisper (requires a live internet connection during the first run).

## Configuration

The following hyper-parameters are exposed and tunable inside the Configuration block of the main notebook file:

| Parameter Variable | Standard Reference Value | Functional System Domain Control |
|---|---|---|
| `MEDICAL_THRESHOLD` | `0.55` | The baseline minimum classification confidence accepted for Type/Status tags without flagging a human override, as well as the early-rejection threshold. |
| `AMBIGUITY_ENTROPY` | `0.85` | Normalized entropy cutoff boundary. Values exceeding this threshold toggle the `review_required` security flag. |
| `MIN_SIDE` / `MAX_SIDE` | `64` / `1280` px | Spatial constraints enforcing valid bounding canvas ranges before triggering an implicit image resize or input rejection. |
| `YOLO_CONF` | `0.25` | Minimum confidence score threshold required for YOLOv8 bounding box generation. |
| `VIDEO_FRAME_STRIDE` | `15` | Frame stepping interval defining how often an active tracking ID profile gets re-evaluated by classification heads under video mode. |
| `CLAP_SR` | `48000` Hz | Target audio sample rate configuration required by the CLAP audio classification framework. |
| `AUDIO_MIN_SECONDS` / `AUDIO_MAX_SECONDS` | `0.5` / `60` s | Valid duration boundaries enforced for input microphone files or raw audio clips. |
| `AUDIO_CONF_FLOOR` | `0.55` | Confidence cutoff floor for CLAP outputs. Scores lower than this boundary return a `non_conclusive` status to avoid forced errors. |
| `VOICE_CONF_OVERRIDE` | `0.85` | Hardcoded confidence fallback value applied when an operator's speech override preempts vision classification (calibrated to preserve `review_required` safety logic). |
| `ENABLE_TTS` | `False` | System switch toggle for activating or deactivating the speech synthesis output module powered by XTTS v2. |
| `SEED` | `42` | Global deterministic configuration value mapped across Python random, NumPy, and PyTorch to guarantee reproducibility. |

## Repository Structure
## Traceability Log Ledger

Every inspection event processed by the application framework (whether approved or rejected by the gatekeeper) is securely appended as a flat entry to `inspecciones_log.csv`. The schema contains the following audit parameters:

- `inspection_id`, `timestamp_utc`, `device_id`
- `tipo_detectado`, `tipo_confianza`, `tipo_entropia`
- `estado_detectado`, `estado_confianza`, `estado_entropia`
- `gatekeeper_passed`, `gatekeeper_score`, `review_required`
- `evidence_sha256`, hash **chained** to the previous record

* **Cryptographic Chaining:** Each record mathematically hashes its local operational data combined with the hash string of the immediate historic line (`previous_record_hash`). Any retroactive manipulation or alteration of an archival log entry breaks the hash link continuity, triggering validation failures during security inspections.

## Known Limitations

- **Zero-Shot Validation Constraints:** The classification architecture (BiomedCLIP/CLAP) operates under zero-shot validation contexts. A comprehensive quantitative test set is still pending to provide matrix metrics differentiating highly similar visual configurations (e.g., distinguishing between a "contaminated surface" vs. "active rust").
- **Sub-String Keyword Dependency:** The verbal voice override relies on strict substring keyword dictionary intersections against Whisper transcriptions. Acoustic distortion or transcription inaccuracies inside Whisper can break text routing silently without raising descriptive exception logs.
- **Video Occlusion ID Resets:** The ByteTrack video inference pipeline can re-index track markers into new unique ID slots when a physical asset suffers short frame occlusions or overlaps, wiping the historically aggregated vision score memory array for that item.
- **Operator Anonymity & Asset Authentication:** The platform does not currently incorporate cryptographic identity verification for active operators or direct parsing validation linking an asset's appearance back to a verified database serial inventory code — see the Ethics section of the technical report.

> 📕 *For a complete analysis regarding edge failure modeling, root-cause hypotheses, and targeted technical mitigations, please refer to the documentation section in the primary technical project report (`docs/M6_Reporte_SATDM.docx`, "Failure cases" section).*

## Ethics and Guardrails

- **Immutable Data Log:** The log structure utilizes an immutable append-only configuration strategy supported by SHA-256 block linking to discourage data falsification.
- **Universal Audit Trail:** Every execution attempt (whether approved or rejected by the gatekeeper) is persistently committed to the system log for security auditing.
- **Automated Escalation Flags:** Unreliable, low-confidence predictions or high-entropy classifications automatically flag the inspection as `review_required=True` to enforce double-check verification workflows.
- **Deepfake Mitigation:** The generative cloning features within the audio stack are **disabled by default** (`ENABLE_TTS=False`) to prevent unauthorized personal voice cloning simulations inside institutional hospital environments.
- **Open Challenge:** The platform does not authenticate the operator or cryptographically link an inspection to the actual serial number of the hardware device — see the technical report for a full analysis.

## Credits

Project developed by **Juan Carlos Aquino Hernández** — Industrial Electromechanics Division, Universidad Tecnológica de Nayarit (**UTNay**) — as part of the TAE-IA course (Cinvestav Guadalajara) and the SATDM research project.
