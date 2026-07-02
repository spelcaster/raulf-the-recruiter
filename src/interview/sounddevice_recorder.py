from __future__ import annotations

import io
import time
from collections.abc import Callable

import numpy as np


SAMPLE_RATE_HZ = 16_000
POLL_INTERVAL_MS = 50


class SoundDeviceRecorder:
    def __init__(self, *, sounddevice_module=None, soundfile_module=None, time_module=None) -> None:
        if sounddevice_module is None:
            import sounddevice as sounddevice_module
        if soundfile_module is None:
            import soundfile as soundfile_module

        self.name = "sounddevice-recorder"
        self._sounddevice = sounddevice_module
        self._soundfile = soundfile_module
        self._time = time_module or time

    def record(self, *, stop_requested: Callable[[], bool], max_duration_seconds: int) -> bytes:
        frames: list[np.ndarray] = []

        def callback(indata, frames_count, time_info, status) -> None:
            del frames_count, time_info, status
            frames.append(indata.copy())

        deadline = self._time.monotonic() + max_duration_seconds
        with self._sounddevice.InputStream(
            samplerate=SAMPLE_RATE_HZ,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            while self._time.monotonic() < deadline:
                if stop_requested():
                    break
                self._sounddevice.sleep(POLL_INTERVAL_MS)

        recording = np.concatenate(frames, axis=0) if frames else np.empty((0, 1), dtype=np.int16)
        wav_buffer = io.BytesIO()
        self._soundfile.write(
            wav_buffer,
            recording,
            SAMPLE_RATE_HZ,
            format="WAV",
            subtype="PCM_16",
        )
        return wav_buffer.getvalue()
