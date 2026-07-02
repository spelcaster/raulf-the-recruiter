from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interview.sounddevice_recorder import SAMPLE_RATE_HZ, SoundDeviceRecorder


class _RecordingInputStream:
    def __init__(self, callback, chunks: list[np.ndarray]) -> None:
        self._callback = callback
        self._chunks = chunks

    def __enter__(self):
        for chunk in self._chunks:
            self._callback(chunk, len(chunk), None, None)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _RecordingSoundDevice:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, object]] = []
        self.sleep_calls: list[int] = []

    def InputStream(self, **kwargs):
        self.calls.append(kwargs)
        return _RecordingInputStream(kwargs["callback"], self._chunks)

    def sleep(self, milliseconds: int) -> None:
        self.sleep_calls.append(milliseconds)


class _RecordingSoundFile:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def write(self, stream, data, sample_rate: int, *, format: str, subtype: str) -> None:
        self.calls.append(
            {
                "data": data.copy(),
                "sample_rate": sample_rate,
                "format": format,
                "subtype": subtype,
            }
        )
        stream.write(b"wav-bytes")


class _MonotonicClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def monotonic(self) -> float:
        return next(self._values)


class SoundDeviceRecorderTests(unittest.TestCase):
    def test_captures_pcm16_mono_wav_until_stop_is_requested(self) -> None:
        chunks = [
            np.array([[1], [2]], dtype=np.int16),
            np.array([[3], [4]], dtype=np.int16),
        ]
        sounddevice_module = _RecordingSoundDevice(chunks)
        soundfile_module = _RecordingSoundFile()
        clock = _MonotonicClock([0.0, 0.01, 0.02])
        recorder = SoundDeviceRecorder(
            sounddevice_module=sounddevice_module,
            soundfile_module=soundfile_module,
            time_module=clock,
        )

        stop_checks = iter([False, True])
        audio = recorder.record(
            stop_requested=lambda: next(stop_checks),
            max_duration_seconds=6 * 60,
        )

        self.assertEqual(audio, b"wav-bytes")
        self.assertEqual(
            sounddevice_module.calls,
            [
                {
                    "samplerate": SAMPLE_RATE_HZ,
                    "channels": 1,
                    "dtype": "int16",
                    "callback": sounddevice_module.calls[0]["callback"],
                }
            ],
        )
        self.assertEqual(sounddevice_module.sleep_calls, [50])
        self.assertEqual(len(soundfile_module.calls), 1)
        self.assertEqual(soundfile_module.calls[0]["sample_rate"], SAMPLE_RATE_HZ)
        self.assertEqual(soundfile_module.calls[0]["format"], "WAV")
        self.assertEqual(soundfile_module.calls[0]["subtype"], "PCM_16")
        np.testing.assert_array_equal(
            soundfile_module.calls[0]["data"],
            np.array([[1], [2], [3], [4]], dtype=np.int16),
        )


if __name__ == "__main__":
    unittest.main()
