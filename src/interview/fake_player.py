from __future__ import annotations


class FakePlayer:
    def __init__(self) -> None:
        self.name = "fake-player"
        self.played: list[bytes] = []

    def play(self, audio: bytes) -> None:
        self.played.append(audio)

