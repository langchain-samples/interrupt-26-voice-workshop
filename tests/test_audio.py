from __future__ import annotations

import unittest

import numpy as np

from voice_workshop.audio import resample_pcm16


class ResamplePCM16Tests(unittest.TestCase):
    def test_empty_audio_is_unchanged(self) -> None:
        self.assertEqual(resample_pcm16(b"", 24_000, 16_000), b"")

    def test_same_rate_is_unchanged(self) -> None:
        pcm = np.arange(100, dtype=np.int16).tobytes()
        self.assertEqual(resample_pcm16(pcm, 24_000, 24_000), pcm)

    def test_resampling_preserves_duration(self) -> None:
        pcm = np.arange(480, dtype=np.int16).tobytes()
        converted = resample_pcm16(pcm, 24_000, 16_000)
        self.assertEqual(len(converted) // 2, 320)

    def test_rejects_invalid_sample_rate(self) -> None:
        with self.assertRaises(ValueError):
            resample_pcm16(b"\x00\x00", 0, 16_000)

    def test_rejects_partial_pcm_sample(self) -> None:
        with self.assertRaises(ValueError):
            resample_pcm16(b"\x00", 24_000, 16_000)


if __name__ == "__main__":
    unittest.main()
