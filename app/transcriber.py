# app/transcriber.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import pipeline

from .exceptions import ModelLoadError, TranscriptionError

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class TranscriptSegment:
    """
    A transcription segment.

    start/end are GLOBAL timestamps in seconds.
    """
    start: float
    end: float
    text: str


# ============================================================
# TRANSCRIBER
# ============================================================

class WhisperTranscriber:
    """
    Local/offline Whisper transcription using Hugging Face
    automatic-speech-recognition pipeline.

    Features
    --------
    - Fully local/offline
    - Hugging Face pipeline
    - CUDA / CPU
    - FP16 / BF16 / FP32
    - Long audio support
    - Automatic chunking
    - 10-second stride/overlap
    - Whisper timestamps
    - Sentence duplicate removal
    - Partial overlap removal
    - No manual timestamp token parsing
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        torch_dtype: str = "float16",
        sample_rate: int = 16000,
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 10.0,
        batch_size: int = 4,
        language: str = "fa",
        task: str = "transcribe",
    ):
        self.model_path = Path(model_path)

        self.device_name = device
        self.sample_rate = sample_rate

        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)

        self.batch_size = max(
            1,
            int(batch_size),
        )

        self.language = language
        self.task = task

        self.dtype = self._resolve_dtype(
            torch_dtype
        )

        self.device = self._resolve_device(
            device
        )

        self.pipe = None

        self._validate_config()
        self._load_pipeline()

    # ========================================================
    # CONFIG
    # ========================================================

    def _validate_config(self) -> None:
        if self.chunk_seconds <= 0:
            raise ModelLoadError(
                "chunk_seconds must be greater than 0."
            )

        if self.overlap_seconds < 0:
            raise ModelLoadError(
                "overlap_seconds cannot be negative."
            )

        if self.overlap_seconds >= self.chunk_seconds:
            raise ModelLoadError(
                "overlap_seconds must be smaller than "
                "chunk_seconds."
            )

        if self.batch_size < 1:
            raise ModelLoadError(
                "batch_size must be >= 1."
            )

        if not self.model_path.exists():
            raise ModelLoadError(
                f"Model path does not exist: "
                f"{self.model_path}"
            )

        if not self.model_path.is_dir():
            raise ModelLoadError(
                f"Model path is not a directory: "
                f"{self.model_path}"
            )

    # ========================================================
    # DTYPE
    # ========================================================

    @staticmethod
    def _resolve_dtype(
        value: str,
    ) -> torch.dtype:
        value = value.lower().strip()

        if value in {
            "float16",
            "fp16",
            "half",
        }:
            return torch.float16

        if value in {
            "bfloat16",
            "bf16",
        }:
            return torch.bfloat16

        if value in {
            "float32",
            "fp32",
            "float",
        }:
            return torch.float32

        raise ModelLoadError(
            f"Unsupported torch dtype: {value}. "
            f"Use float16, bfloat16 or float32."
        )

    # ========================================================
    # DEVICE
    # ========================================================

    @staticmethod
    def _resolve_device(
        device: str,
    ) -> int | str:
        """
        transformers.pipeline expects:

            device=0       -> first CUDA GPU
            device=1       -> second CUDA GPU
            device="cpu"   -> CPU

        """

        device = device.lower().strip()

        if device == "cuda":
            if not torch.cuda.is_available():
                raise ModelLoadError(
                    "CUDA was requested but no CUDA GPU "
                    "is available."
                )

            return 0

        if device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise ModelLoadError(
                    "CUDA was requested but CUDA is "
                    "unavailable."
                )

            try:
                index = int(
                    device.split(":", 1)[1]
                )
            except ValueError as exc:
                raise ModelLoadError(
                    f"Invalid CUDA device: {device}"
                ) from exc

            if index >= torch.cuda.device_count():
                raise ModelLoadError(
                    f"CUDA device {index} does not exist. "
                    f"Available GPUs: "
                    f"{torch.cuda.device_count()}"
                )

            return index

        if device == "cpu":
            return "cpu"

        raise ModelLoadError(
            f"Unsupported device: {device}. "
            f"Use cuda, cuda:N or cpu."
        )

    # ========================================================
    # LOAD PIPELINE
    # ========================================================

    def _load_pipeline(self) -> None:
        logger.info(
            "Loading local Whisper pipeline..."
        )

        logger.info(
            "Model path: %s",
            self.model_path,
        )

        try:
            # ------------------------------------------------
            # IMPORTANT:
            #
            # These environment variables should normally
            # also be set in main.py / Dockerfile, but we
            # explicitly request local_files_only here.
            # ------------------------------------------------

            self.pipe = pipeline(
                task="automatic-speech-recognition",

                model=str(
                    self.model_path
                ),

                device=self.device,

                torch_dtype=self.dtype,

                model_kwargs={
                    "local_files_only": True,
                },
            )

            # ------------------------------------------------
            # GPU information
            # ------------------------------------------------

            if self.device != "cpu":
                gpu_index = int(
                    self.device
                )

                gpu_name = (
                    torch.cuda.get_device_name(
                        gpu_index
                    )
                )

                total_memory = (
                    torch.cuda
                    .get_device_properties(
                        gpu_index
                    )
                    .total_memory
                    / (1024**3)
                )

                logger.info(
                    "GPU: %s",
                    gpu_name,
                )

                logger.info(
                    "GPU memory: %.2f GB",
                    total_memory,
                )

            logger.info(
                "Whisper pipeline loaded successfully."
            )

            logger.info(
                "dtype: %s",
                self.dtype,
            )

            logger.info(
                "device: %s",
                self.device,
            )

            logger.info(
                "batch_size: %d",
                self.batch_size,
            )

        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise ModelLoadError(
                "GPU out of memory while loading "
                "the Whisper model."
            ) from exc

        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Whisper pipeline: {exc}"
            ) from exc

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize text ONLY for comparison.

        The actual transcript is not modified by this method.
        """

        if not text:
            return ""

        text = str(text)

        replacements = {
            "ي": "ی",
            "ى": "ی",
            "ئ": "ی",
            "ك": "ک",
            "ة": "ه",
            "ۀ": "ه",
            "ؤ": "و",
            "إ": "ا",
            "أ": "ا",
            "ٱ": "ا",
        }

        for old, new in replacements.items():
            text = text.replace(
                old,
                new,
            )

        # Zero-width characters
        text = text.replace(
            "\u200c",
            " ",
        )

        text = text.replace(
            "\u200d",
            "",
        )

        text = text.replace(
            "\ufeff",
            "",
        )

        # Arabic/Persian diacritics
        text = re.sub(
            r"[\u0610-\u061A"
            r"\u064B-\u065F"
            r"\u0670"
            r"\u06D6-\u06ED]",
            "",
            text,
        )

        # Whitespace
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip().lower()

    @classmethod
    def _words(
        cls,
        text: str,
    ) -> list[str]:
        normalized = cls._normalize_text(
            text
        )

        if not normalized:
            return []

        return normalized.split()

    # ========================================================
    # SENTENCE SPLIT
    # ========================================================

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        if not text:
            return []

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not text:
            return []

        parts = re.split(
            r"(?<=[.!?؟؛])\s+",
            text,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # ========================================================
    # SEQUENCE SIMILARITY
    # ========================================================

    @classmethod
    def _sequence_similarity(
        cls,
        text_a: str,
        text_b: str,
    ) -> float:
        """
        Conservative word-sequence similarity.
        """

        a = cls._words(text_a)
        b = cls._words(text_b)

        if not a or not b:
            return 0.0

        if a == b:
            return 1.0

        # ----------------------------------------------------
        # Shorter sequence completely exists in longer one.
        # ----------------------------------------------------

        if len(a) >= 3 and len(b) >= 3:
            if len(a) <= len(b):
                shorter = a
                longer = b
            else:
                shorter = b
                longer = a

            n = len(shorter)

            for i in range(
                len(longer) - n + 1
            ):
                if (
                    longer[i:i + n]
                    == shorter
                ):
                    return 0.95

        # ----------------------------------------------------
        # Ordered common sequence.
        # ----------------------------------------------------

        common = 0

        for word in a:
            if word in b:
                common += 1

        denominator = max(
            len(a),
            len(b),
        )

        if denominator == 0:
            return 0.0

        return common / denominator

    # ========================================================
    # TIMESTAMP OVERLAP
    # ========================================================

    @staticmethod
    def _temporal_overlap(
        a: TranscriptSegment,
        b: TranscriptSegment,
    ) -> float:
        return max(
            0.0,
            min(a.end, b.end)
            - max(a.start, b.start),
        )

    @staticmethod
    def _duration(
        segment: TranscriptSegment,
    ) -> float:
        return max(
            0.0,
            segment.end - segment.start,
        )

    # ========================================================
    # DUPLICATE SEGMENT
    # ========================================================

    @classmethod
    def _is_duplicate_segment(
        cls,
        previous: TranscriptSegment,
        current: TranscriptSegment,
    ) -> bool:
        """
        Determine whether two timestamped chunks represent
        the same speech.

        Timestamp overlap is required.

        This prevents something like:

            "سلام"
            ...

            120 seconds later:
            "سلام"

        from being considered a duplicate.
        """

        previous_norm = cls._normalize_text(
            previous.text
        )

        current_norm = cls._normalize_text(
            current.text
        )

        if not previous_norm or not current_norm:
            return False

        previous_words = previous_norm.split()
        current_words = current_norm.split()

        min_words = min(
            len(previous_words),
            len(current_words),
        )

        # ----------------------------------------------------
        # Temporal overlap
        # ----------------------------------------------------

        overlap = cls._temporal_overlap(
            previous,
            current,
        )

        if overlap <= 0:
            return False

        previous_duration = cls._duration(
            previous
        )

        current_duration = cls._duration(
            current
        )

        min_duration = min(
            previous_duration,
            current_duration,
        )

        if min_duration <= 0:
            return False

        overlap_ratio = (
            overlap / min_duration
        )

        # ----------------------------------------------------
        # Exact match
        # ----------------------------------------------------

        if previous_norm == current_norm:

            # For very short phrases be stricter.
            if min_words <= 3:
                return overlap_ratio >= 0.50

            return overlap_ratio >= 0.20

        # ----------------------------------------------------
        # Similar text
        # ----------------------------------------------------

        similarity = (
            cls._sequence_similarity(
                previous.text,
                current.text,
            )
        )

        # Long text + strong overlap
        if (
            min_words >= 5
            and similarity >= 0.90
            and overlap_ratio >= 0.35
        ):
            return True

        # Very long text + somewhat lower similarity
        if (
            min_words >= 8
            and similarity >= 0.82
            and overlap_ratio >= 0.60
        ):
            return True

        return False

    # ========================================================
    # REMOVE DUPLICATE SENTENCES
    # ========================================================

    @classmethod
    def _remove_duplicate_sentences(
        cls,
        previous_text: str,
        current_text: str,
    ) -> str:
        """
        Remove duplicate sentences from the beginning
        of current_text.

        Example:

            previous:
                این یک تست است. امروز هوا خوب است.

            current:
                امروز هوا خوب است. حالا ادامه می‌دهیم.

        result:

            حالا ادامه می‌دهیم.
        """

        previous_sentences = (
            cls._split_sentences(
                previous_text
            )
        )

        current_sentences = (
            cls._split_sentences(
                current_text
            )
        )

        if not previous_sentences:
            return current_text.strip()

        if not current_sentences:
            return current_text.strip()

        previous_tail = previous_sentences[-6:]

        remove_count = 0

        for current_sentence in current_sentences:

            current_norm = cls._normalize_text(
                current_sentence
            )

            if not current_norm:
                continue

            matched = False

            for previous_sentence in reversed(
                previous_tail
            ):
                previous_norm = (
                    cls._normalize_text(
                        previous_sentence
                    )
                )

                if not previous_norm:
                    continue

                # Exact normalized sentence.
                if (
                    current_norm
                    == previous_norm
                ):
                    matched = True
                    break

                current_words = cls._words(
                    current_sentence
                )

                previous_words = cls._words(
                    previous_sentence
                )

                min_words = min(
                    len(current_words),
                    len(previous_words),
                )

                if min_words < 4:
                    continue

                similarity = (
                    cls._sequence_similarity(
                        current_sentence,
                        previous_sentence,
                    )
                )

                if similarity >= 0.92:
                    matched = True
                    break

            if matched:
                remove_count += 1
            else:
                break

        if remove_count == 0:
            return current_text.strip()

        if remove_count >= len(
            current_sentences
        ):
            return ""

        return " ".join(
            sentence
            for sentence in current_sentences[
                remove_count:
            ]
        ).strip()

    # ========================================================
    # REMOVE PARTIAL WORD OVERLAP
    # ========================================================

    @classmethod
    def _remove_partial_word_overlap(
        cls,
        previous_text: str,
        current_text: str,
        max_words: int = 30,
    ) -> str:
        """
        Remove repeated suffix/prefix words.

        Example:

            previous:
                ... امروز درباره سیستم صحبت می‌کنیم

            current:
                امروز درباره سیستم صحبت می‌کنیم و
                بعد به بخش دوم می‌رسیم

        becomes:

            و بعد به بخش دوم می‌رسیم
        """

        previous_words = cls._words(
            previous_text
        )

        current_words_normalized = cls._words(
            current_text
        )

        current_words_original = (
            current_text.strip().split()
        )

        if not previous_words:
            return current_text.strip()

        if not current_words_normalized:
            return current_text.strip()

        max_overlap = min(
            max_words,
            len(previous_words),
            len(current_words_normalized),
        )

        found = 0

        # At least 3 words must overlap.
        for size in range(
            max_overlap,
            2,
            -1,
        ):
            previous_tail = (
                previous_words[-size:]
            )

            current_head = (
                current_words_normalized[
                    :size
                ]
            )

            if (
                previous_tail
                == current_head
            ):
                found = size
                break

        if found == 0:
            return current_text.strip()

        remaining = (
            current_words_original[
                found:
            ]
        )

        return " ".join(
            remaining
        ).strip()

    # ========================================================
    # REMOVE TEXT OVERLAP
    # ========================================================

    @classmethod
    def _remove_text_overlap(
        cls,
        previous_text: str,
        current_text: str,
    ) -> str:
        if not current_text:
            return ""

        current_text = (
            current_text.strip()
        )

        previous_text = (
            previous_text.strip()
        )

        if not previous_text:
            return current_text

        # ----------------------------------------------------
        # First remove complete repeated sentences.
        # ----------------------------------------------------

        current_text = (
            cls._remove_duplicate_sentences(
                previous_text,
                current_text,
            )
        )

        if not current_text:
            return ""

        # ----------------------------------------------------
        # Then remove partial word overlap.
        # ----------------------------------------------------

        current_text = (
            cls._remove_partial_word_overlap(
                previous_text,
                current_text,
            )
        )

        return current_text.strip()

    # ========================================================
    # MERGE PIPELINE CHUNKS
    # ========================================================

    @classmethod
    def _merge_segments(
        cls,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        """
        Merge pipeline chunks while removing duplicated
        speech caused by stride/overlap.
        """

        if not segments:
            return []

        ordered = sorted(
            segments,
            key=lambda x: (
                x.start,
                x.end,
            ),
        )

        merged: list[
            TranscriptSegment
        ] = []

        # Whisper stride is normally 10 sec.
        # Use a little wider window to account for
        # timestamp variations.
        lookback_seconds = 15.0

        temporal_tolerance = 0.75

        for current in ordered:

            current_text = (
                current.text.strip()
            )

            if not current_text:
                continue

            # ------------------------------------------------
            # Find recent segments.
            # ------------------------------------------------

            recent: list[
                TranscriptSegment
            ] = []

            for previous in reversed(
                merged
            ):

                if (
                    current.start
                    - previous.end
                    > lookback_seconds
                ):
                    break

                if (
                    previous.end
                    >= current.start
                    - temporal_tolerance
                ):
                    recent.append(previous)

            # ------------------------------------------------
            # Complete duplicate
            # ------------------------------------------------

            duplicate = False

            for previous in recent:
                if cls._is_duplicate_segment(
                    previous,
                    current,
                ):
                    logger.debug(
                        "Duplicate removed: "
                        "[%.2f - %.2f] %s",
                        current.start,
                        current.end,
                        current.text,
                    )

                    duplicate = True
                    break

            if duplicate:
                continue

            # ------------------------------------------------
            # Partial overlap
            # ------------------------------------------------

            modified_text = current_text

            for previous in recent:

                overlap = cls._temporal_overlap(
                    previous,
                    current,
                )

                if overlap <= 0:
                    continue

                previous_duration = (
                    cls._duration(previous)
                )

                current_duration = (
                    cls._duration(current)
                )

                min_duration = min(
                    previous_duration,
                    current_duration,
                )

                if min_duration <= 0:
                    continue

                overlap_ratio = (
                    overlap / min_duration
                )

                # Conservative threshold.
                if overlap_ratio < 0.10:
                    continue

                modified_text = (
                    cls._remove_text_overlap(
                        previous.text,
                        modified_text,
                    )
                )

                if not modified_text:
                    break

            if not modified_text:
                continue

            merged.append(
                TranscriptSegment(
                    start=current.start,
                    end=current.end,
                    text=modified_text,
                )
            )

        return merged

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    @staticmethod
    def _cleanup_final_text(
        text: str,
    ) -> str:
        if not text:
            return ""

        # Spaces
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Spaces before punctuation
        text = re.sub(
            r"\s+([،؛.!?؟])",
            r"\1",
            text,
        )

        # Missing spaces after punctuation
        text = re.sub(
            r"([،؛!?؟])(?=\S)",
            r"\1 ",
            text,
        )

        return text.strip()

    # ========================================================
    # PIPELINE RESULT PARSER
    # ========================================================

    @staticmethod
    def _extract_chunks(
        result,
    ) -> list[dict]:
        """
        Extract pipeline chunks safely.
        """

        if not isinstance(
            result,
            dict,
        ):
            return []

        chunks = result.get(
            "chunks",
            [],
        )

        if not isinstance(
            chunks,
            list,
        ):
            return []

        return [
            chunk
            for chunk in chunks
            if isinstance(
                chunk,
                dict,
            )
        ]

    # ========================================================
    # TRANSCRIBE ONE FILE
    # ========================================================

    def transcribe(
        self,
        audio_path: Path,
        chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
    ) -> str:
        """
        Transcribe a potentially very long audio/media file.

        The Hugging Face ASR pipeline handles chunking and
        stride internally.

        Parameters
        ----------
        audio_path:
            Path to audio/video/media file.

        chunk_seconds:
            Chunk length. Defaults to configured value.

        overlap_seconds:
            Stride/overlap. Defaults to configured value.
        """

        if self.pipe is None:
            raise TranscriptionError(
                "Whisper pipeline is not initialized."
            )

        audio_path = Path(
            audio_path
        )

        if not audio_path.exists():
            raise TranscriptionError(
                f"Audio file does not exist: "
                f"{audio_path}"
            )

        if not audio_path.is_file():
            raise TranscriptionError(
                f"Audio path is not a file: "
                f"{audio_path}"
            )

        chunk_seconds = (
            self.chunk_seconds
            if chunk_seconds is None
            else float(chunk_seconds)
        )

        overlap_seconds = (
            self.overlap_seconds
            if overlap_seconds is None
            else float(overlap_seconds)
        )

        if chunk_seconds <= 0:
            raise TranscriptionError(
                "chunk_seconds must be greater than 0."
            )

        if overlap_seconds < 0:
            raise TranscriptionError(
                "overlap_seconds cannot be negative."
            )

        if overlap_seconds >= chunk_seconds:
            raise TranscriptionError(
                "overlap_seconds must be smaller than "
                "chunk_seconds."
            )

        logger.info(
            "Starting transcription: %s",
            audio_path,
        )

        logger.info(
            "chunk_length_s=%.2f",
            chunk_seconds,
        )

        logger.info(
            "stride_length_s=%.2f",
            overlap_seconds,
        )

        logger.info(
            "batch_size=%d",
            self.batch_size,
        )

        try:
            # ------------------------------------------------
            # Pipeline inference
            # ------------------------------------------------
            #
            # Passing the file path directly allows the
            # transformers ASR pipeline to decode the input.
            #
            # chunk_length_s:
            #   chunk size
            #
            # stride_length_s:
            #   overlap on both sides / stride mechanism
            #
            # return_timestamps=True:
            #   gives result["chunks"] with timestamps.
            # ------------------------------------------------

            result = self.pipe(
                str(audio_path),

                chunk_length_s=chunk_seconds,

                stride_length_s=overlap_seconds,

                batch_size=self.batch_size,

                return_timestamps=True,
            )

        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise TranscriptionError(
                "GPU out of memory during transcription. "
                "Try reducing batch_size."
            ) from exc

        except RuntimeError as exc:

            if (
                "out of memory"
                in str(exc).lower()
            ):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise TranscriptionError(
                    "GPU out of memory during transcription. "
                    "Try reducing batch_size."
                ) from exc

            raise TranscriptionError(
                f"Whisper pipeline inference failed: "
                f"{exc}"
            ) from exc

        except Exception as exc:
            raise TranscriptionError(
                f"Whisper pipeline inference failed: "
                f"{exc}"
            ) from exc

        # ----------------------------------------------------
        # Extract chunks
        # ----------------------------------------------------

        chunks = self._extract_chunks(
            result
        )

        # ----------------------------------------------------
        # Normal pipeline result
        # ----------------------------------------------------

        raw_segments: list[
            TranscriptSegment
        ] = []

        for chunk in chunks:

            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            timestamp = chunk.get(
                "timestamp"
            )

            if (
                not isinstance(
                    timestamp,
                    (tuple, list),
                )
                or len(timestamp) != 2
            ):
                continue

            start = timestamp[0]
            end = timestamp[1]

            # Some versions may return None.
            if start is None:
                start = 0.0

            if end is None:
                # We cannot know exact end here.
                # Use chunk start + configured chunk length.
                end = (
                    float(start)
                    + chunk_seconds
                )

            try:
                start = float(start)
                end = float(end)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if end <= start:
                continue

            raw_segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                )
            )

        # ----------------------------------------------------
        # If timestamps were unavailable, use complete text.
        # ----------------------------------------------------

        if not raw_segments:

            full_text = ""

            if isinstance(
                result,
                dict,
            ):
                full_text = str(
                    result.get(
                        "text",
                        "",
                    )
                ).strip()

            if full_text:
                logger.warning(
                    "Pipeline returned text but no "
                    "timestamped chunks. "
                    "Returning full text without "
                    "timestamp-based deduplication."
                )

                return self._cleanup_final_text(
                    full_text
                )

            logger.warning(
                "No transcription text was produced "
                "for %s",
                audio_path,
            )

            return ""

        logger.info(
            "Pipeline produced %d raw chunks.",
            len(raw_segments),
        )

        # ----------------------------------------------------
        # Remove duplicate speech generated by stride.
        # ----------------------------------------------------

        merged_segments = (
            self._merge_segments(
                raw_segments
            )
        )

        logger.info(
            "Segments after deduplication: %d",
            len(merged_segments),
        )

        # ----------------------------------------------------
        # Build final transcript.
        # ----------------------------------------------------

        final_parts = [
            segment.text.strip()
            for segment in merged_segments
            if segment.text.strip()
        ]

        final_text = " ".join(
            final_parts
        )

        final_text = (
            self._cleanup_final_text(
                final_text
            )
        )

        logger.info(
            "Transcription finished: %s "
            "(%d characters)",
            audio_path,
            len(final_text),
        )

        return final_text