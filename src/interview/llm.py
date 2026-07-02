from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    name: str

    def next_turn(self, *, seed_instruction: str, history: list[str]) -> str:
        """Return the next interviewer utterance for the current session."""
