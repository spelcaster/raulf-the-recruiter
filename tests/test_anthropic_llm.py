from __future__ import annotations

import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interview.anthropic_llm import AnthropicLLM, MODEL_NAME, fake_anthropic_message


class _RecordingMessagesAPI:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return fake_anthropic_message(self._response_text)


class _RecordingAnthropicClient:
    def __init__(self, response_text: str) -> None:
        self.messages = _RecordingMessagesAPI(response_text)


class AnthropicLLMTests(unittest.TestCase):
    def test_sends_seed_as_system_prompt_and_maps_turn_history_in_order(self) -> None:
        client = _RecordingAnthropicClient("Interviewer reply")
        llm = AnthropicLLM("test-key", client=client)

        reply = llm.next_turn(
            seed_instruction="Conduza uma entrevista para engenheiro backend",
            history=["Opening question", "My answer", "Second question", "Another answer"],
        )

        self.assertEqual(reply, "Interviewer reply")
        self.assertEqual(len(client.messages.calls), 1)
        self.assertEqual(
            client.messages.calls[0],
            {
                "model": MODEL_NAME,
                "max_tokens": 512,
                "system": (
                    "You are the Interviewer in an English-language mock interview.\n"
                    "Use the Seed Instruction below as the scenario and constraints for the interview.\n"
                    "Conduct the interview in English even if the Seed Instruction is written in another language.\n"
                    "Stay in character as the interviewer.\n"
                    "Never correct the speaker's English during the interview.\n"
                    "Save all feedback for the evaluation phase after the session.\n\n"
                    "Seed Instruction:\nConduza uma entrevista para engenheiro backend"
                ),
                "messages": [
                    {"role": "assistant", "content": "Opening question"},
                    {"role": "user", "content": "My answer"},
                    {"role": "assistant", "content": "Second question"},
                    {"role": "user", "content": "Another answer"},
                ],
            },
        )

    def test_raises_when_anthropic_returns_no_text_blocks(self) -> None:
        class EmptyMessagesAPI:
            def create(self, **kwargs):
                return type("Message", (), {"content": [type("Block", (), {"type": "tool_use"})()]})()

        client = type("Client", (), {"messages": EmptyMessagesAPI()})()
        llm = AnthropicLLM("test-key", client=client)

        with self.assertRaisesRegex(RuntimeError, "did not include any text content"):
            llm.next_turn(seed_instruction="Seed", history=[])


if __name__ == "__main__":
    unittest.main()
