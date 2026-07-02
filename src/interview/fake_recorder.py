from __future__ import annotations


class FakeRecorder:
    def __init__(self, recordings: list[bytes] | None = None) -> None:
        self._recordings = list(recordings or [b"default recording"])
        self.name = "fake-recorder"

    def record(self, *, stop_requested, max_duration_seconds: int) -> bytes:
        del max_duration_seconds
        while not stop_requested():
            pass
        if self._recordings:
            return self._recordings.pop(0)
        return b"default recording"
