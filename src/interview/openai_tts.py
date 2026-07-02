from __future__ import annotations

from openai import OpenAI


MODEL_NAME = "gpt-4o-mini-tts"


class OpenAITTS:
    def __init__(self, api_key: str, *, voice: str, client: OpenAI | None = None) -> None:
        self.name = MODEL_NAME
        self.voice = voice
        self._client = client or OpenAI(api_key=api_key)

    def synthesize(self, text: str) -> bytes:
        response = self._client.audio.speech.create(
            model=MODEL_NAME,
            voice=self.voice,
            input=text,
            response_format="wav",
        )
        return response.read()
