from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interview.environment import Environment
from interview.session_runner import SessionRunner


class _RecordingTTS:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.voice = "test-voice"
        self.name = "recording-tts"
        self.calls: list[str] = []
        self._output_dir = output_dir

    def synthesize(self, text: str) -> bytes:
        if self._output_dir is not None:
            session_dirs = [path for path in self._output_dir.iterdir() if path.is_dir()]
            assert len(session_dirs) == 1, "session dir should exist before synthesize"
            interviewer_path = session_dirs[0] / f"interviewer_{len(self.calls) + 1:03d}.txt"
            assert interviewer_path.exists(), f"{interviewer_path.name} should exist before synthesize"
            assert interviewer_path.read_text(encoding="utf-8").strip() == text
        self.calls.append(text)
        return f"WAV:{text}".encode("utf-8")


class SessionRunnerLLMContractTests(unittest.TestCase):
    def test_rebuilds_seed_and_history_for_each_llm_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = Environment.build_fake(
                llm_responses=["Opening question", "Follow-up question"],
                transcripts=["My answer"],
            )
            runner = SessionRunner(
                environment=environment,
                stdin=io.StringIO("t\na\nq\n"),
                stdout=io.StringIO(),
                output_dir=Path(tmp),
            )

            exit_code = runner.start(seed_instruction="Faça a entrevista para vaga de backend")

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                environment.llm.calls,
                [
                    {
                        "seed_instruction": "Faça a entrevista para vaga de backend",
                        "history": [],
                    },
                    {
                        "seed_instruction": "Faça a entrevista para vaga de backend",
                        "history": ["Opening question", "My answer"],
                    },
                ],
            )

    def test_saves_and_autoplays_interviewer_audio_and_replays_latest_audio_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            environment = Environment.build_fake(
                llm_responses=["Opening question", "Follow-up question"],
                transcripts=["Accepted answer"],
            )
            runner = SessionRunner(
                environment=environment,
                stdin=io.StringIO("a\nt\na\na\nq\n"),
                stdout=io.StringIO(),
                output_dir=output_dir,
            )

            exit_code = runner.start(seed_instruction="Seed")

            self.assertEqual(exit_code, 0)
            session_dir = next(output_dir.iterdir())
            self.assertEqual(
                (session_dir / "interviewer_001.wav").read_bytes(),
                b"WAV:Opening question",
            )
            self.assertEqual(
                (session_dir / "interviewer_002.wav").read_bytes(),
                b"WAV:Follow-up question",
            )
            self.assertEqual(
                environment.player.played,
                [
                    b"WAV:Opening question",
                    b"WAV:Opening question",
                    b"WAV:Follow-up question",
                    b"WAV:Follow-up question",
                ],
            )

    def test_persists_interviewer_text_before_tts_synthesis_begins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            environment = Environment.build_fake(
                llm_responses=["Opening question"],
                transcripts=[],
            )
            environment.tts = _RecordingTTS(output_dir)
            runner = SessionRunner(
                environment=environment,
                stdin=io.StringIO("q\n"),
                stdout=io.StringIO(),
                output_dir=output_dir,
            )

            exit_code = runner.start(seed_instruction="Seed")

            self.assertEqual(exit_code, 0)
            session_dir = next(output_dir.iterdir())
            self.assertEqual(environment.tts.calls, ["Opening question"])
            self.assertEqual((session_dir / "interviewer_001.wav").read_bytes(), b"WAV:Opening question")


if __name__ == "__main__":
    unittest.main()
