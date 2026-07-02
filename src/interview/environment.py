from __future__ import annotations

import os
from typing import Callable, Protocol

from interview.anthropic_llm import AnthropicLLM
from interview.fake_llm import FakeLLM
from interview.fake_player import FakePlayer
from interview.fake_recorder import FakeRecorder
from interview.fake_stt import FakeSTT
from interview.fake_tts import FakeTTS
from interview.llm import LLM


class TTS(Protocol):
    name: str
    voice: str

    def synthesize(self, text: str) -> bytes: ...


class STT(Protocol):
    name: str

    def transcribe(self, recording: bytes) -> str: ...


class Recorder(Protocol):
    name: str

    def record(self, *, stop_requested: Callable[[], bool], max_duration_seconds: int) -> bytes: ...


class Player(Protocol):
    name: str

    def play(self, audio: bytes) -> None: ...


class Environment:
    def __init__(
        self,
        *,
        llm: LLM,
        tts: TTS,
        stt: STT,
        recorder: Recorder,
        player: Player,
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
        voice: str = "fake-voice",
    ) -> "Environment":
        return cls(
            llm=FakeLLM(llm_responses),
            tts=FakeTTS(voice=voice),
            stt=FakeSTT(transcripts),
            recorder=FakeRecorder(recordings),
            player=FakePlayer(),
        )

    @classmethod
    def build_anthropic(cls, *, voice: str) -> "Environment":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY must be set before starting a session")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set before starting a session")

        from interview.openai_stt import OpenAISTT
        from interview.openai_tts import OpenAITTS
        from interview.sounddevice_player import SoundDevicePlayer
        from interview.sounddevice_recorder import SoundDeviceRecorder

        return cls(
            llm=AnthropicLLM(api_key),
            tts=OpenAITTS(openai_api_key, voice=voice),
            stt=OpenAISTT(openai_api_key),
            recorder=SoundDeviceRecorder(),
            player=SoundDevicePlayer(),
        )
