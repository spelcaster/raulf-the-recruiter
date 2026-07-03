from __future__ import annotations

from types import SimpleNamespace

from anthropic import Anthropic


MODEL_NAME = "claude-opus-4-8"
MAX_TOKENS = 512


class AnthropicLLM:
    def __init__(self, api_key: str, *, client: Anthropic | None = None) -> None:
        self.name = MODEL_NAME
        self._client = client or Anthropic(api_key=api_key)

    def next_turn(self, *, seed_instruction: str, history: list[str]) -> str:
        return self._create_text_response(
            system=self._system_prompt(seed_instruction),
            messages=self._messages_from_history(history),
        )

    def evaluate_session(self, *, prompt: str) -> str:
        return self._create_text_response(
            system=(
                "You are an English coach evaluating a completed mock interview.\n"
                "Follow the user's requested markdown structure exactly and keep the feedback qualitative."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

    def _system_prompt(self, seed_instruction: str) -> str:
        return (
            "You are the Interviewer in an English-language mock interview.\n"
            "Use the Seed Instruction below as the scenario and constraints for the interview.\n"
            "Conduct the interview in English even if the Seed Instruction is written in another language.\n"
            "Stay in character as the interviewer.\n"
            "Never correct the speaker's English during the interview.\n"
            "Save all feedback for the evaluation phase after the session.\n\n"
            f"Seed Instruction:\n{seed_instruction.strip()}"
        )

    def _messages_from_history(self, history: list[str]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for index, turn in enumerate(history):
            role = "assistant" if index % 2 == 0 else "user"
            messages.append({"role": role, "content": turn})
        return messages

    def _create_text_response(self, *, system: str, messages: list[dict[str, str]]) -> str:
        response = self._client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
        if text:
            return text
        raise RuntimeError("Anthropic response did not include any text content")


def fake_anthropic_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
