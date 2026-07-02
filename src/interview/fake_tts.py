from __future__ import annotations


class FakeTTS:
    def __init__(self, voice: str = "fake-voice") -> None:
        self.voice = voice
        self.name = "fake-tts"

    def synthesize(self, text: str) -> bytes:
        return text.encode("utf-8")

