from __future__ import annotations

from io import BytesIO


class SoundDevicePlayer:
    def __init__(self, *, sounddevice_module=None, soundfile_module=None) -> None:
        self.name = "sounddevice-player"
        if sounddevice_module is None:
            import sounddevice as sounddevice_module
        if soundfile_module is None:
            import soundfile as soundfile_module
        self._sounddevice = sounddevice_module
        self._soundfile = soundfile_module

    def play(self, audio: bytes) -> None:
        pcm, sample_rate = self._soundfile.read(BytesIO(audio), dtype="float32")
        self._sounddevice.play(pcm, sample_rate)
        self._sounddevice.wait()
