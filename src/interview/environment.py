from __future__ import annotations

from interview.fake_llm import FakeLLM
from interview.fake_player import FakePlayer
from interview.fake_recorder import FakeRecorder
from interview.fake_stt import FakeSTT
from interview.fake_tts import FakeTTS


class Environment:
    def __init__(
        self,
        *,
        llm: FakeLLM,
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

