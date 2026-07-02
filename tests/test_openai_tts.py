from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interview.openai_tts import MODEL_NAME, OpenAITTS
from interview.sounddevice_player import SoundDevicePlayer


class _RecordingSpeechAPI:
    def __init__(self, audio_bytes: bytes) -> None:
        self._audio_bytes = audio_bytes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(read=lambda: self._audio_bytes)


class _RecordingOpenAIClient:
    def __init__(self, audio_bytes: bytes) -> None:
        self.audio = SimpleNamespace(speech=_RecordingSpeechAPI(audio_bytes))


class _RecordingSoundFile:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def read(self, stream, *, dtype: str):
        self.calls.append({"audio": stream.read(), "dtype": dtype})
        return ["pcm-samples"], 24_000


class _RecordingSoundDevice:
    def __init__(self) -> None:
        self.play_calls: list[tuple[object, int]] = []
        self.wait_calls = 0

    def play(self, pcm, sample_rate: int) -> None:
        self.play_calls.append((pcm, sample_rate))

    def wait(self) -> None:
        self.wait_calls += 1


class OpenAITTSTests(unittest.TestCase):
    def test_requests_wav_audio_with_selected_voice(self) -> None:
        client = _RecordingOpenAIClient(b"wav-bytes")
        tts = OpenAITTS("test-key", voice="nova", client=client)

        audio = tts.synthesize("Opening question")

        self.assertEqual(audio, b"wav-bytes")
        self.assertEqual(
            client.audio.speech.calls,
            [
                {
                    "model": MODEL_NAME,
                    "voice": "nova",
                    "input": "Opening question",
                    "response_format": "wav",
                }
            ],
        )

    def test_sounddevice_player_decodes_wav_bytes_and_waits_for_playback(self) -> None:
        soundfile_module = _RecordingSoundFile()
        sounddevice_module = _RecordingSoundDevice()
        player = SoundDevicePlayer(
            sounddevice_module=sounddevice_module,
            soundfile_module=soundfile_module,
        )

        player.play(b"wav-bytes")

        self.assertEqual(
            soundfile_module.calls,
            [{"audio": b"wav-bytes", "dtype": "float32"}],
        )
        self.assertEqual(sounddevice_module.play_calls, [(["pcm-samples"], 24_000)])
        self.assertEqual(sounddevice_module.wait_calls, 1)


if __name__ == "__main__":
    unittest.main()
