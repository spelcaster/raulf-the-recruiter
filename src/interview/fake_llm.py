from __future__ import annotations


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.name = "fake-llm"

    def next_turn(self, *, seed_instruction: str, history: list[str]) -> str:
        if self._responses:
            return self._responses.pop(0)
        return f"Follow-up for: {seed_instruction}"

