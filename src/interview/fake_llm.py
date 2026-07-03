from __future__ import annotations


class FakeLLM:
    def __init__(self, responses: list[str], evaluation_responses: list[str] | None = None) -> None:
        self._responses = list(responses)
        self._evaluation_responses = list(evaluation_responses or [])
        self.name = "fake-llm"
        self.calls: list[dict[str, list[str] | str]] = []
        self.evaluation_calls: list[dict[str, str]] = []

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

    def evaluate_session(self, *, prompt: str) -> str:
        self.evaluation_calls.append({"prompt": prompt})
        if self._evaluation_responses:
            return self._evaluation_responses.pop(0)
        return "# Language\nCEFR: B2\n\n# Content\nStrongest answer: n/a\n\n# Overall\nPractice next."
