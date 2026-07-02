from __future__ import annotations


class FakeSTT:
    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self.name = "fake-stt"

    def transcribe(self, recording: bytes) -> str:
        if self._transcripts:
            return self._transcripts.pop(0)
        return recording.decode("utf-8")

