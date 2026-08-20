"""Thread-safe local PCM16 microphone and speaker utilities for Gemini Live."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import AsyncIterator, Callable
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

CHANNELS = 1
DTYPE = "int16"
CHUNK_MS = 20
MAX_MIC_CHUNKS = 100
MAX_SPEAKER_SECONDS = 10


def resample_pcm16(data: bytes, source_rate: int, target_rate: int) -> bytes:
    """Resample mono PCM16 bytes while preserving their approximate duration."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if len(data) % 2:
        raise ValueError("PCM16 data must contain complete 16-bit samples")
    if not data or source_rate == target_rate:
        return data

    samples = np.frombuffer(data, dtype=np.int16)
    divisor = gcd(source_rate, target_rate)
    converted = resample_poly(
        samples.astype(np.float32),
        target_rate // divisor,
        source_rate // divisor,
    )
    return np.clip(np.rint(converted), -32768, 32767).astype(np.int16).tobytes()


def _frames_per_chunk(sample_rate: int) -> int:
    return int(sample_rate * CHUNK_MS / 1_000)


class MicStream:
    """An async PCM16 stream backed by the default input device."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MAX_MIC_CHUNKS)
        self._stream: sd.RawInputStream | None = None

    def start(self) -> None:
        if self._stream is not None:
            return
        self._loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            del frames, time_info, status
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            try:
                loop.call_soon_threadsafe(self._put_fresh, bytes(indata))
            except RuntimeError:
                return

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=_frames_per_chunk(self.sample_rate),
            callback=callback,
        )
        self._stream.start()

    def _put_fresh(self, data: bytes) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(data)

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            yield await self._queue.get()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        self._loop = None
        if stream is not None:
            stream.stop()
            stream.close()


class SpeakerStream:
    """A bounded PCM16 output buffer that can be cleared immediately on barge-in."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self._max_bytes = sample_rate * 2 * MAX_SPEAKER_SECONDS
        self._chunks: deque[bytes] = deque()
        self._queued_bytes = 0
        self._current = bytearray()
        self._lock = threading.Lock()
        self._stream: sd.RawOutputStream | None = None
        self._on_played: Callable[[bytes], None] | None = None

    def set_played_callback(self, callback: Callable[[bytes], None] | None) -> None:
        self._on_played = callback

    def start(self) -> None:
        if self._stream is not None:
            return

        def callback(outdata, frames, time_info, status) -> None:  # noqa: ANN001
            del frames, time_info, status
            needed = len(outdata)
            with self._lock:
                while len(self._current) < needed and self._chunks:
                    chunk = self._chunks.popleft()
                    self._queued_bytes -= len(chunk)
                    self._current.extend(chunk)
                played = bytes(self._current[:needed])
                del self._current[:needed]
                outdata[:] = played + b"\x00" * (needed - len(played))
            callback_fn = self._on_played
            if callback_fn is not None and played:
                callback_fn(played)

        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=_frames_per_chunk(self.sample_rate),
            callback=callback,
        )
        self._stream.start()

    def write(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            self._chunks.append(bytes(data))
            self._queued_bytes += len(data)
            while self._queued_bytes + len(self._current) > self._max_bytes:
                if self._chunks:
                    dropped = self._chunks.popleft()
                    self._queued_bytes -= len(dropped)
                else:
                    self._current.clear()
                    break

    def buffered_bytes(self) -> int:
        with self._lock:
            return self._queued_bytes + len(self._current)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._queued_bytes = 0
            self._current.clear()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        self.clear()
