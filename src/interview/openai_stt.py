from __future__ import annotations

from io import BytesIO

from openai import OpenAI


MODEL_NAME = "gpt-4o-transcribe"


class OpenAISTT:
    def __init__(self, api_key: str, *, client: OpenAI | None = None) -> None:
        self.name = MODEL_NAME
        self._client = client or OpenAI(api_key=api_key)

    def transcribe(self, recording: bytes) -> str:
        response = self._client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=("speaker.wav", BytesIO(recording), "audio/wav"),
            response_format="text",
        )
        if isinstance(response, str):
            return response
        if hasattr(response, "text"):
            return response.text
        raise RuntimeError("OpenAI transcription response did not include text")
