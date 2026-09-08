FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# Python
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        tini && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Offline HuggingFace
# ============================================================

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Never attempt telemetry
ENV DO_NOT_TRACK=1
ENV HF_HUB_DISABLE_TELEMETRY=1

# ============================================================
# Python Runtime
# ============================================================

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ============================================================
# pip
# ============================================================

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    --upgrade \
    pip \
    setuptools \
    wheel

# ============================================================
# PyTorch CUDA 12.8
# ============================================================

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    torch==2.7.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ============================================================
# Python dependencies
# ============================================================

COPY requirement.txt .

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    -r requirement.txt

# ============================================================
# Application
# ============================================================

COPY app ./app

# Model mount point + temporary STT directory
RUN mkdir -p \
    /models/nezamisafa/whisper-persian-v4 \
    /tmp/stt

# ============================================================
# Security
# ============================================================

RUN useradd \
    --create-home \
    --shell /usr/sbin/nologin \
    appuser

RUN chown -R appuser:appuser \
    /app \
    /tmp/stt

USER appuser

# ============================================================
# Runtime
# ============================================================

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]