from __future__ import annotations

import os

from interview.anthropic_llm import AnthropicLLM
from interview.fake_llm import FakeLLM
from interview.fake_player import FakePlayer
from interview.fake_recorder import FakeRecorder
from interview.fake_stt import FakeSTT
from interview.fake_tts import FakeTTS
from interview.llm import LLM


class Environment:
    def __init__(
        self,
        *,
        llm: LLM,
        tts: FakeTTS,
        stt: FakeSTT,
        recorder: FakeRecorder,
        player: FakePlayer,
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.stt = stt
        self.recorder = recorder
        self.player = player

    @classmethod
    def build_fake(
        cls,
        *,
        llm_responses: list[str],
        transcripts: list[str],
        recordings: list[bytes] | None = None,
    ) -> "Environment":
        return cls(
            llm=FakeLLM(llm_responses),
            tts=FakeTTS(),
            stt=FakeSTT(transcripts),
            recorder=FakeRecorder(recordings),
            player=FakePlayer(),
        )

    @classmethod
    def build_anthropic(cls) -> "Environment":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY must be set before starting a session")

        return cls(
            llm=AnthropicLLM(api_key),
            tts=FakeTTS(),
            stt=FakeSTT(["Example transcript"]),
            recorder=FakeRecorder(),
            player=FakePlayer(),
        )
