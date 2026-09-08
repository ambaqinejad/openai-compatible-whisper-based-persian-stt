# app/audio.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from .exceptions import AudioProcessingError


logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    index: int
    start: float
    end: float
    audio: np.ndarray


@dataclass
class AudioInfo:
    duration: float | None
    sample_rate: int
    channels: int


class AudioDecoder:

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 10.0,
    ):

        self.sample_rate = sample_rate
        self.channels = channels

        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

        if overlap_seconds >= chunk_seconds:
            raise ValueError(
                "overlap_seconds must be smaller than chunk_seconds"
            )

        self.step_seconds = (
            chunk_seconds - overlap_seconds
        )

    def get_info(self, path: Path) -> AudioInfo:

        try:
            with av.open(str(path)) as container:

                stream = next(
                    (
                        s
                        for s in container.streams
                        if s.type == "audio"
                    ),
                    None,
                )

                if stream is None:
                    raise AudioProcessingError(
                        "No audio stream found"
                    )

                duration = None

                if stream.duration is not None:
                    duration = float(
                        stream.duration * stream.time_base
                    )

                return AudioInfo(
                    duration=duration,
                    sample_rate=stream.rate or 0,
                    channels=stream.channels or 0,
                )

        except av.error.FFmpegError as exc:

            raise AudioProcessingError(
                f"Could not inspect audio file: {exc}"
            ) from exc

    def iter_chunks(
        self,
        path: Path,
    ) -> Iterator[AudioChunk]:

        """
        Streaming audio decoder.

        Important:
        The entire audio file is never loaded into RAM.

        Chunks:

            chunk = 30 sec
            overlap = 10 sec
            step = 20 sec

        Therefore:

            chunk 0: 0   -> 30
            chunk 1: 20  -> 50
            chunk 2: 40  -> 70
            ...
        """

        try:
            container = av.open(str(path))

        except av.error.FFmpegError as exc:

            raise AudioProcessingError(
                f"Could not open audio file: {exc}"
            ) from exc

        try:

            audio_stream = next(
                (
                    s
                    for s in container.streams
                    if s.type == "audio"
                ),
                None,
            )

            if audio_stream is None:
                raise AudioProcessingError(
                    "Input file does not contain an audio stream"
                )

            resampler = av.audio.resampler.AudioResampler(
                format="fltp",
                layout="mono",
                rate=self.sample_rate,
            )

            buffer = np.empty(
                0,
                dtype=np.float32,
            )

            chunk_samples = int(
                self.chunk_seconds * self.sample_rate
            )

            step_samples = int(
                self.step_seconds * self.sample_rate
            )

            overlap_samples = int(
                self.overlap_seconds * self.sample_rate
            )

            chunk_index = 0

            for frame in container.decode(
                audio=audio_stream.index
            ):

                try:
                    frames = resampler.resample(frame)

                except Exception as exc:

                    raise AudioProcessingError(
                        f"Audio resampling failed: {exc}"
                    ) from exc

                if not isinstance(frames, list):
                    frames = [frames]

                for resampled in frames:

                    if resampled is None:
                        continue

                    data = resampled.to_ndarray()

                    if data.ndim == 2:
                        data = data[0]

                    data = np.asarray(
                        data,
                        dtype=np.float32,
                    )

                    buffer = np.concatenate(
                        (buffer, data)
                    )

                    while len(buffer) >= chunk_samples:

                        chunk = buffer[
                            :chunk_samples
                        ].copy()

                        start = (
                            chunk_index
                            * self.step_seconds
                        )

                        end = (
                            start
                            + self.chunk_seconds
                        )

                        yield AudioChunk(
                            index=chunk_index,
                            start=start,
                            end=end,
                            audio=chunk,
                        )

                        # Keep overlap.
                        buffer = buffer[
                            step_samples:
                        ]

                        chunk_index += 1

            # Flush decoder/resampler
            try:

                flushed = resampler.resample(None)

                if flushed is not None:

                    if not isinstance(flushed, list):
                        flushed = [flushed]

                    for resampled in flushed:

                        if resampled is None:
                            continue

                        data = resampled.to_ndarray()

                        if data.ndim == 2:
                            data = data[0]

                        data = np.asarray(
                            data,
                            dtype=np.float32,
                        )

                        buffer = np.concatenate(
                            (buffer, data)
                        )

            except Exception:
                logger.warning(
                    "Could not flush audio resampler",
                    exc_info=True,
                )

            # Last partial chunk.
            if len(buffer) > 0:

                start = (
                    chunk_index
                    * self.step_seconds
                )

                end = (
                    start
                    + len(buffer) / self.sample_rate
                )

                # Ignore extremely tiny tails.
                if len(buffer) >= int(
                    0.25 * self.sample_rate
                ):

                    yield AudioChunk(
                        index=chunk_index,
                        start=start,
                        end=end,
                        audio=buffer.copy(),
                    )

        except av.error.FFmpegError as exc:

            raise AudioProcessingError(
                f"Audio decoding failed: {exc}"
            ) from exc

        finally:

            container.close()