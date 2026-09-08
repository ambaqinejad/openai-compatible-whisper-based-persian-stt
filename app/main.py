# app/main.py

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse

from .config import get_settings
from .exceptions import (
    AudioProcessingError,
    AudioTooLargeError,
    AuthenticationError,
    InvalidAudioError,
    ModelLoadError,
    TranscriptionError,
)
from .logging_config import configure_logging
from .transcriber import WhisperTranscriber


settings = get_settings()

configure_logging(settings.log_level)

logger = logging.getLogger(__name__)


transcriber: WhisperTranscriber | None = None

transcription_semaphore = asyncio.Semaphore(
    settings.max_concurrent_transcriptions
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global transcriber

    logger.info(
        "Starting %s v%s",
        settings.app_name,
        settings.app_version,
    )

    logger.info(
        "Offline mode enabled"
    )

    transcriber = WhisperTranscriber(
        model_path=settings.model_path,
        device=settings.device,
        torch_dtype=settings.torch_dtype,
        sample_rate=settings.sample_rate,
    )

    yield

    logger.info(
        "Shutting down"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


# ============================================================
# Authentication
# ============================================================

async def authenticate(
    authorization: str | None = Header(
        default=None
    ),
):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
        )

    expected = (
        f"Bearer {settings.api_key}"
    )

    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


# ============================================================
# Health
# ============================================================

@app.get(
    "/health",
)
async def health():
    return {
        "status": "ok",
        "model": "nezamisafa/whisper-persian-v4",
        "offline": True,
        "device": settings.device,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get(
    "/ready",
)
async def ready():

    if transcriber is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ready",
        "model": str(
            settings.model_path
        ),
    }


# ============================================================
# OpenAI compatible model endpoint
# ============================================================

@app.get(
    "/v1/models",
    dependencies=[Depends(authenticate)],
)
async def list_models():

    return {
        "object": "list",
        "data": [
            {
                "id": "nezamisafa/whisper-persian-v4",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


# ============================================================
# OpenAI-compatible transcription endpoint
# ============================================================

@app.post(
    "/v1/audio/transcriptions",
    # dependencies=[Depends(authenticate)],
)
async def create_transcription(
    file: UploadFile = File(...),

    model: str = Form(
        default="nezamisafa/whisper-persian-v4"
    ),

    language: str | None = Form(
        default="fa"
    ),

    response_format: str = Form(
        default="json"
    ),

    temperature: float = Form(
        default=0.0
    ),
):

    request_id = uuid.uuid4().hex

    logger.info(
        "[%s] Transcription request: %s",
        request_id,
        file.filename,
    )

    if transcriber is None:

        raise HTTPException(
            status_code=503,
            detail="STT model is not ready",
        )

    if model not in {
        "nezamisafa/whisper-persian-v4",
        "whisper-1",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported model: {model}"
            ),
        )

    if response_format not in {
        "json",
        "text",
        "verbose_json",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported response_format"
            ),
        )

    # --------------------------------------------------------
    # Temporary file
    # --------------------------------------------------------

    suffix = Path(
        file.filename or "audio"
    ).suffix

    if not suffix:
        suffix = ".audio"

    tmp_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            dir=settings.upload_dir,
            delete=False,
        ) as tmp:

            tmp_path = Path(
                tmp.name
            )

            total = 0

            max_bytes = (
                settings.max_upload_size_mb
                * 1024
                * 1024
            )

            while True:

                data = await file.read(
                    1024 * 1024
                )

                if not data:
                    break

                total += len(data)

                if total > max_bytes:

                    raise AudioTooLargeError(
                        "Uploaded file exceeds "
                        f"{settings.max_upload_size_mb} MB"
                    )

                tmp.write(data)

        if total == 0:

            raise InvalidAudioError(
                "Uploaded file is empty"
            )

        # ----------------------------------------------------
        # Serialize GPU inference
        # ----------------------------------------------------

        async with transcription_semaphore:

            logger.info(
                "[%s] Starting transcription",
                request_id,
            )

            text = await asyncio.to_thread(
                transcriber.transcribe,
                tmp_path,
                settings.chunk_seconds,
                settings.overlap_seconds,
            )

        logger.info(
            "[%s] Transcription completed",
            request_id,
        )

        if response_format == "text":

            return text

        if response_format == "verbose_json":

            return {
                "task": "transcribe",
                "language": language or "fa",
                "duration": None,
                "text": text,
            }

        return {
            "text": text
        }

    except AudioTooLargeError as exc:

        raise HTTPException(
            status_code=413,
            detail=str(exc),
        ) from exc

    except InvalidAudioError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except AudioProcessingError as exc:

        logger.exception(
            "[%s] Audio processing error",
            request_id,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except TranscriptionError as exc:

        logger.exception(
            "[%s] Transcription error",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "[%s] Unexpected error",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal transcription error"
            ),
        ) from exc

    finally:

        if tmp_path is not None:

            try:
                tmp_path.unlink(
                    missing_ok=True
                )

            except Exception:

                logger.warning(
                    "[%s] Could not delete "
                    "temporary file %s",
                    request_id,
                    tmp_path,
                )

        await file.close()