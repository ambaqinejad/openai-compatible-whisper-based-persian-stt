# Persian Whisper STT API

A production-oriented **local and offline Persian Speech-to-Text API** built with:

* [FastAPI](https://fastapi.tiangolo.com/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
* Whisper
* CUDA / NVIDIA GPU
* Docker & Docker Compose
* Python

The API is designed around the [`nezamisafa/whisper-persian-v4`](https://huggingface.co/nezamisafa/whisper-persian-v4) model and is optimized for **Persian audio transcription**, including long audio files.

It can be used as a standalone REST API or as a backend for applications such as Open WebUI.

---

## Features

* 🇮🇷 Persian speech-to-text
* 🚀 NVIDIA CUDA GPU acceleration
* ⚡ FP16 / BF16 / FP32 support
* 📴 Fully offline inference
* 🔒 No runtime connection to Hugging Face
* 🎵 Supports common audio/media formats through the Transformers audio pipeline
* ⏱️ Long audio support
* ✂️ Automatic audio chunking
* 🔄 10-second stride/overlap
* 🧩 Batch inference
* 🕒 Whisper timestamps
* 🧹 Duplicate sentence removal
* 🧹 Partial overlap removal
* 📝 Clean final Persian transcript
* 🌐 FastAPI REST API
* 🔑 API key authentication
* 🐳 Docker support
* 🐳 Docker Compose support
* 💾 Local model support
* 📊 Detailed logging
* 🩺 Health and readiness endpoints
* 🔌 OpenAI-compatible transcription endpoint

---

# Architecture

The project is intentionally simple:

```text
Client
  │
  │ HTTP
  ▼
FastAPI
  │
  ├── Authentication
  │
  ├── Upload handling
  │
  └── Transcription
          │
          ▼
    Hugging Face Pipeline
          │
          ▼
   Whisper Persian Model
          │
          ▼
    CUDA / NVIDIA GPU
```

For long audio:

```text
Long Audio
    │
    ▼
Whisper Pipeline
    │
    ├── Chunk 1
    ├── Chunk 2
    ├── Chunk 3
    ├── ...
    └── Chunk N
          │
          ▼
    Timestamped chunks
          │
          ▼
    Duplicate removal
          │
          ▼
    Overlap removal
          │
          ▼
    Final Persian transcript
```

---

# Model

This project uses:

**nezamisafa/whisper-persian-v4**

Model page:

https://huggingface.co/nezamisafa/whisper-persian-v4

The model is a Persian fine-tuned Whisper model intended for Persian speech recognition.

The model is expected to exist locally.

For example:

```text
persian-stt/
├── app/
├── model/
│   └── whisper-persian-v4/
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

The application does **not** download the model during runtime.

---

# Offline Mode

The application is designed to run without internet access after the model and dependencies have been installed.

The following environment variables should be enabled:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

The pipeline also uses:

```python
model_kwargs={
    "local_files_only": True,
}
```

Therefore, the application will not attempt to download missing model files from Hugging Face during inference.

---

# Project Structure

```text
persian-stt/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── exceptions.py
│   ├── logging_config.py
│   ├── models.py
│   ├── audio.py
│   └── transcriber.py
│
├── model/
│   └── whisper-persian-v4/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env
└── README.md
```

---

# Requirements

## Hardware

For GPU inference:

* NVIDIA GPU
* NVIDIA Container Toolkit
* Docker with NVIDIA GPU support

A GPU with sufficient VRAM is recommended for the Whisper model.

For example:

```text
NVIDIA RTX 4080 / 4080 Super
```

can be used for local inference.

CPU inference is also possible, but will generally be significantly slower.

---

# Python Dependencies

Example dependencies:

```text
fastapi==0.116.1
uvicorn[standard]==0.35.0
python-multipart==0.0.20
pydantic==2.11.7
pydantic-settings==2.10.1
transformers==4.52.4
accelerate==1.8.1
safetensors==0.5.3
numpy==2.2.6
av==14.4.0
psutil==7.0.0
```

PyTorch can be installed separately with the CUDA build appropriate for your environment.

For CUDA 12.8:

```bash
python3 -m pip install \
    torch==2.7.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

---

# Local Model Setup

Download the model once on a machine with internet access.

The model directory should contain the required Hugging Face files.

For example:

```text
model/whisper-persian-v4/
├── config.json
├── generation_config.json
├── model.safetensors
├── preprocessor_config.json
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── ...
```

Then copy the complete directory to the server.

Example:

```text
./model/whisper-persian-v4
```

The application can then run completely offline.

---

# Configuration

Configuration is controlled through environment variables.

Example `.env`:

```env
APP_NAME=Persian Whisper STT
APP_VERSION=1.0.0

HOST=0.0.0.0
PORT=8000

API_KEY=change-me

MODEL_PATH=/models/whisper-persian-v4

SAMPLE_RATE=16000
CHANNELS=1

CHUNK_SECONDS=30
OVERLAP_SECONDS=10

MAX_UPLOAD_SIZE_MB=4096

UPLOAD_DIR=/tmp/stt

DEVICE=cuda
TORCH_DTYPE=float16

LANGUAGE=fa
TASK=transcribe

MAX_CONCURRENT_TRANSCRIPTIONS=1

NUM_BEAMS=1
TEMPERATURE=0.0
CONDITION_ON_PREV_TOKENS=true

LOG_LEVEL=INFO

HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

---

# Chunking

Long audio is handled through the Hugging Face ASR pipeline.

The default configuration is:

```text
chunk_length_s = 30
stride_length_s = 10
```

This means the audio is processed using approximately 30-second chunks with a 10-second overlap/stride.

For example:

```text
Audio
│
├────────────── 30s ──────────────┤
       ├────────────── 30s ──────────────┤
              ├────────────── 30s ──────────────┤
```

The overlap is important because speech can cross chunk boundaries.

The application then uses Whisper timestamps and text comparison to reduce repeated speech caused by overlapping chunks.

---

# Duplicate Removal

Chunking with overlap can produce output such as:

```text
Chunk 1:
امروز می‌خواهیم درباره سیستم تبدیل صوت صحبت کنیم.

Chunk 2:
درباره سیستم تبدیل صوت صحبت کنیم. سپس وارد بخش دوم می‌شویم.
```

Without post-processing:

```text
امروز می‌خواهیم درباره سیستم تبدیل صوت صحبت کنیم.
درباره سیستم تبدیل صوت صحبت کنیم.
سپس وارد بخش دوم می‌شویم.
```

The application attempts to produce:

```text
امروز می‌خواهیم درباره سیستم تبدیل صوت صحبت کنیم.
سپس وارد بخش دوم می‌شویم.
```

Duplicate detection considers:

* Timestamp overlap
* Exact normalized text
* Word-level similarity
* Sentence similarity
* Partial suffix/prefix overlap

Text normalization is used only for comparison. The original transcript text is preserved whenever possible.

---

# Batch Processing

The Transformers pipeline supports batch inference.

Example:

```python
result = self.pipe(
    str(audio_path),
    chunk_length_s=30,
    stride_length_s=10,
    batch_size=4,
    return_timestamps=True,
)
```

The batch size can be increased if GPU memory allows it.

For example:

```env
BATCH_SIZE=8
```

Higher batch sizes can improve throughput but require more VRAM.

If you encounter:

```text
CUDA out of memory
```

reduce the batch size.

For example:

```text
8 → 4 → 2 → 1
```

---

# Docker

The recommended deployment method is Docker.

Example `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        tini && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    torch==2.7.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    -r requirements.txt

COPY app ./app

RUN mkdir -p /tmp/stt

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# Docker Compose

Example:

```yaml
services:

  persian-stt:
    build:
      context: .
      dockerfile: Dockerfile

    container_name: persian-stt

    restart: unless-stopped

    ports:
      - "8000:8000"

    env_file:
      - .env

    volumes:
      - ./model/whisper-persian-v4:/models/whisper-persian-v4:ro
      - stt_tmp:/tmp/stt

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities:
                - gpu

volumes:
  stt_tmp:
```

Start the service:

```bash
docker compose up -d --build
```

View logs:

```bash
docker compose logs -f persian-stt
```

Stop:

```bash
docker compose down
```

---

# Verify NVIDIA GPU

Before starting the application, verify that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 nvidia-smi
```

You should see your NVIDIA GPU and its VRAM information.

If this command fails, fix Docker/NVIDIA GPU support before troubleshooting the application.

---

# API

The main transcription endpoint is:

```text
POST /v1/audio/transcriptions
```

This endpoint follows the familiar OpenAI transcription API structure.

---

# Authentication

The API can use a Bearer API key.

Example:

```http
Authorization: Bearer change-me
```

With curl:

```bash
curl \
  -X POST \
  http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer change-me" \
  -F "file=@audio.mp3"
```

---

# Transcribe an MP3

Example:

```bash
curl \
  -X POST \
  http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer change-me" \
  -F "file=@sample.mp3"
```

Example response:

```json
{
  "text": "سلام. این یک نمونه تبدیل گفتار فارسی به متن است."
}
```

---

# Transcribe Long Audio

Long files are supported.

For example:

```bash
curl \
  -X POST \
  http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer change-me" \
  -F "file=@two_hour_audio.mp3"
```

The application does not require you to manually split the file.

The Transformers pipeline handles the chunking:

```text
2 hour audio
      │
      ▼
30 second chunks
      │
      ▼
10 second stride
      │
      ▼
Whisper
      │
      ▼
timestamped chunks
      │
      ▼
deduplication
      │
      ▼
final transcript
```

---

# Supported Media

The underlying Transformers audio pipeline can decode common media formats supported by the installed audio backend.

Examples include:

```text
.mp3
.wav
.m4a
.flac
.ogg
.mp4
.mkv
```

For video files, the audio stream can be extracted/decoded through the media backend.

For maximum compatibility with problematic files, converting the audio to a standard format such as WAV or MP3 may be useful.

---

# Health Check

The service provides a health endpoint:

```text
GET /health
```

Example:

```bash
curl http://localhost:8000/health
```

Example:

```json
{
  "status": "ok"
}
```

---

# Readiness Check

Use:

```text
GET /ready
```

This endpoint can be used to determine whether the Whisper pipeline is initialized and ready to process requests.

Example:

```bash
curl http://localhost:8000/ready
```

---

# Models Endpoint

The API also exposes:

```text
GET /v1/models
```

Example:

```bash
curl \
  http://localhost:8000/v1/models \
  -H "Authorization: Bearer change-me"
```

---

# Python Client

A simple Python client:

```python
import requests


url = "http://localhost:8000/v1/audio/transcriptions"

headers = {
    "Authorization": "Bearer change-me",
}

with open("audio.mp3", "rb") as audio:

    response = requests.post(
        url,
        headers=headers,
        files={
            "file": (
                "audio.mp3",
                audio,
                "audio/mpeg",
            )
        },
        timeout=3600,
    )

response.raise_for_status()

print(response.json())
```

---

# JavaScript Client

Example using `fetch`:

```javascript
const form = new FormData();

form.append(
    "file",
    document.querySelector("#audio").files[0]
);

const response = await fetch(
    "http://localhost:8000/v1/audio/transcriptions",
    {
        method: "POST",

        headers: {
            "Authorization": "Bearer change-me"
        },

        body: form
    }
);

const result = await response.json();

console.log(result.text);
```

---

# Open WebUI Integration

The API is designed to expose an OpenAI-style transcription endpoint:

```text
/v1/audio/transcriptions
```

Therefore it can be used as an external STT backend by applications that support OpenAI-compatible audio transcription APIs.

For example:

```text
Open WebUI
     │
     │ HTTP
     ▼
Persian STT API
     │
     ▼
Whisper Persian v4
     │
     ▼
NVIDIA GPU
```

If Open WebUI and the STT API are running on different machines, make sure:

1. The STT API listens on:

```text
0.0.0.0
```

2. Port `8000` is exposed.

3. Windows Firewall allows inbound TCP traffic on port `8000`.

4. The Open WebUI container can reach the IP address of the STT machine.

5. Do not use `localhost` or `127.0.0.1` from inside the Open WebUI container to reach a service running on another machine.

For example, if the STT machine has:

```text
192.168.1.50
```

the container should connect to:

```text
http://192.168.1.50:8000
```

rather than:

```text
http://localhost:8000
```

---

# Logging

The application provides detailed logs for transcription.

Example:

```text
INFO - Starting transcription: /tmp/stt/audio.mp3

INFO - chunk_length_s=30.00
INFO - stride_length_s=10.00
INFO - batch_size=4

INFO - Pipeline returned 24 chunks.

INFO - Processing chunk 1/24 |
timestamp=(0.0, 30.0) |
text=سلام امروز می‌خواهیم...

INFO - Chunk 1/24 processed successfully |
0.00s -> 30.00s |
chars=187

INFO - Processing chunk 2/24 |
timestamp=(20.0, 50.0) |
text=می‌خواهیم درباره سیستم...

INFO - Chunk 2/24 processed successfully |
20.00s -> 50.00s |
chars=201

...

INFO - Pipeline produced 24 raw chunks.
INFO - Segments after deduplication: 19

INFO - Transcription finished:
.../audio.mp3
(4217 characters)
```

This is particularly useful for debugging long transcription jobs.

---

# Performance

Performance depends on:

* GPU model
* GPU VRAM
* audio duration
* audio encoding
* speech density
* batch size
* chunk size
* model precision
* CPU performance
* disk speed

For NVIDIA GPUs, FP16 is generally a good starting point:

```env
DEVICE=cuda
TORCH_DTYPE=float16
```

For GPUs with suitable BF16 support:

```env
TORCH_DTYPE=bfloat16
```

For CPU:

```env
DEVICE=cpu
TORCH_DTYPE=float32
```

---

# GPU Memory

Batch size has a direct impact on GPU memory consumption.

If:

```text
batch_size=8
```

causes:

```text
CUDA out of memory
```

try:

```text
batch_size=4
```

then:

```text
batch_size=2
```

or:

```text
batch_size=1
```

A practical starting point for a high-end NVIDIA GPU is:

```text
FP16
batch_size=4
chunk_length_s=30
stride_length_s=10
```

Then increase the batch size if VRAM allows.

---

# Error Handling

The application defines custom exceptions for different failure categories:

```text
STTException
├── AudioProcessingError
├── AudioTooLargeError
├── InvalidAudioError
├── TranscriptionError
├── ModelLoadError
└── AuthenticationError
```

GPU out-of-memory errors are detected and converted into application-level transcription/model errors.

When an inference OOM occurs, CUDA cache cleanup is attempted:

```python
torch.cuda.empty_cache()
```

---

# Security

The API supports API-key authentication.

Do not use:

```env
API_KEY=change-me
```

in a production deployment.

Generate a strong random key and configure:

```env
API_KEY=<your-secret-key>
```

If the API is exposed outside a trusted local network, place it behind HTTPS and an appropriate reverse proxy.

---

# Offline Deployment

A completely offline deployment can be prepared on a machine with internet access.

Install/download:

```text
Python dependencies
PyTorch
Transformers
Whisper model
Docker images
```

Then transfer the required files/images to the offline machine.

At runtime:

```text
Internet
   X
   │
   │
   ▼
Persian STT API
   │
   ├── Local model
   ├── Local Python packages
   └── Local CUDA runtime
```

No model download is required during transcription.

---

# Example Workflow

## 1. Clone the project

```bash
git clone <YOUR_REPOSITORY_URL>
cd persian-stt
```

## 2. Put the model in the model directory

```text
model/
└── whisper-persian-v4/
```

## 3. Configure `.env`

```env
MODEL_PATH=/models/whisper-persian-v4
DEVICE=cuda
TORCH_DTYPE=float16
CHUNK_SECONDS=30
OVERLAP_SECONDS=10
```

## 4. Build

```bash
docker compose build
```

## 5. Start

```bash
docker compose up -d
```

## 6. Check logs

```bash
docker compose logs -f
```

## 7. Check health

```bash
curl http://localhost:8000/health
```

## 8. Transcribe

```bash
curl \
  -X POST \
  http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer change-me" \
  -F "file=@audio.mp3"
```

---

# Troubleshooting

## CUDA is not available

Check:

```bash
nvidia-smi
```

Then check Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 nvidia-smi
```

---

## Model not found

Check:

```text
MODEL_PATH
```

Inside the container:

```bash
docker exec -it persian-stt ls -lah /models/whisper-persian-v4
```

The directory must contain the complete model.

---

## Hugging Face connection error

Make sure:

```env
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

and that the model is completely available locally.

The pipeline also uses:

```python
local_files_only=True
```

---

## CUDA out of memory

Reduce:

```text
batch_size
```

For example:

```text
8 → 4 → 2 → 1
```

Also make sure other GPU applications are not consuming VRAM.

---

## Open WebUI cannot connect

If the API works from the host but not from Open WebUI, remember that:

```text
localhost
```

inside a Docker container refers to the container itself.

Use the actual IP address of the STT server:

```text
http://192.168.x.x:8000
```

Also verify Windows Firewall and network connectivity.

---

# Development

Run without Docker:

```bash
python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000
```

For development with automatic reload:

```bash
python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

---

# API Example

Basic request:

```http
POST /v1/audio/transcriptions
Authorization: Bearer <API_KEY>
Content-Type: multipart/form-data

file=<audio file>
```

Response:

```json
{
  "text": "متن نهایی تبدیل شده از گفتار فارسی."
}
```

---

# Design Goals

This project focuses on four main goals:

### 1. Local inference

The model runs locally using the user's own GPU.

### 2. Offline operation

No runtime dependency on Hugging Face or external APIs.

### 3. Long-form transcription

Large audio files can be processed using automatic chunking and overlap.

### 4. Clean Persian output

Overlapping Whisper chunks are post-processed to reduce duplicated sentences and repeated words.

---

# Limitations

The quality of transcription depends on:

* audio quality
* microphone quality
* background noise
* speaker overlap
* accents
* recording conditions
* model limitations

The overlap-removal logic is intentionally conservative because aggressive deduplication can accidentally remove legitimate repeated speech.

For example:

```text
"بله بله بله"
```

should not automatically be treated as a duplicate merely because the same words appear multiple times.

---

# License

This project is distributed under the license specified by the repository.

The Whisper Persian model has its own license and terms.

Please review the model's license before redistributing the model:

https://huggingface.co/nezamisafa/whisper-persian-v4

---

# Acknowledgements

This project builds on the following technologies:

* Whisper
* Hugging Face Transformers
* PyTorch
* FastAPI
* NVIDIA CUDA
* Docker

Special thanks to the authors of the Persian Whisper model:

**nezamisafa / whisper-persian-v4**

---

# Roadmap

Potential future improvements:

* [ ] Streaming transcription
* [ ] WebSocket transcription
* [ ] Speaker diarization
* [ ] Word-level timestamps
* [ ] SRT subtitle generation
* [ ] VTT subtitle generation
* [ ] TXT export
* [ ] JSON transcript export
* [ ] Queue-based asynchronous jobs
* [ ] Redis task queue
* [ ] Multi-GPU inference
* [ ] Prometheus metrics
* [ ] Automatic language detection
* [ ] Better Persian punctuation restoration
* [ ] Advanced Persian text normalization

---

# Contributing

Contributions, bug reports and improvements are welcome.

Before submitting a pull request:

1. Test the API.
2. Test a short audio file.
3. Test a long audio file.
4. Check GPU memory usage.
5. Verify offline operation.
6. Include relevant logs for bugs.

---

# License of This Repository

See the repository license file:

```text
LICENSE
```

---

## Quick Start

For the impatient:

```bash
git clone <YOUR_REPOSITORY_URL>
cd persian-stt

# Put the model here:
# ./model/whisper-persian-v4/

docker compose up -d --build

curl http://localhost:8000/health

curl \
  -X POST \
  http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer change-me" \
  -F "file=@audio.mp3"
```

That's it.

The audio is sent to the local FastAPI service, processed by the local Persian Whisper model on the NVIDIA GPU, chunked automatically for long files, deduplicated, and returned as a Persian transcript.
