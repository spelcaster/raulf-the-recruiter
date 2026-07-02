from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interview.openai_stt import MODEL_NAME, OpenAISTT


class _RecordingTranscriptionsAPI:
    def __init__(self, transcript: str) -> None:
        self._transcript = transcript
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._transcript


class _RecordingOpenAIClient:
    def __init__(self, transcript: str) -> None:
        self.audio = SimpleNamespace(transcriptions=_RecordingTranscriptionsAPI(transcript))


class OpenAISTTTests(unittest.TestCase):
    def test_requests_wav_transcription_with_gpt_4o_transcribe(self) -> None:
        client = _RecordingOpenAIClient("Transcribed answer")
        stt = OpenAISTT("test-key", client=client)

        transcript = stt.transcribe(b"wav-bytes")

        self.assertEqual(transcript, "Transcribed answer")
        self.assertEqual(len(client.audio.transcriptions.calls), 1)
        call = client.audio.transcriptions.calls[0]
        self.assertEqual(call["model"], MODEL_NAME)
        filename, payload, content_type = call["file"]
        self.assertEqual(filename, "speaker.wav")
        self.assertEqual(payload.read(), b"wav-bytes")
        self.assertEqual(content_type, "audio/wav")


if __name__ == "__main__":
    unittest.main()
