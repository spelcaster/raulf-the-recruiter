from __future__ import annotations


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.name = "fake-llm"
        self.calls: list[dict[str, list[str] | str]] = []

    def next_turn(self, *, seed_instruction: str, history: list[str]) -> str:
        self.calls.append(
            {
                "seed_instruction": seed_instruction,
                "history": list(history),
            }
        )
        if self._responses:
            return self._responses.pop(0)
        return f"Follow-up for: {seed_instruction}"
