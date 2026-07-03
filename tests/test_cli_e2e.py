from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interview.cli import main
from interview.environment import Environment


class CliEndToEndTests(unittest.TestCase):
    def test_evaluate_last_rebuilds_session_transcript_and_saves_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            session_dir = self._write_session_fixture(
                output_dir,
                ended_cleanly=False,
                seed_instruction="Interview me for a backend role",
                turns=[
                    ("interviewer", 1, "Tell me about yourself."),
                    ("speaker", 1, "I work with Python and distributed systems."),
                    ("interviewer", 2, "Describe a production incident."),
                    ("speaker", 2, "I traced a queue backlog to a bad retry policy."),
                ],
            )
            environment = Environment.build_fake(
                llm_responses=[],
                transcripts=[],
                evaluation_responses=[
                    "# Language\nCEFR: B2\n\n# Content\nStrongest answer: the incident example.\n\n# Overall\nPractice articles and filler words."
                ],
            )

            exit_code = main(
                ["evaluate", "--last", "--output-dir", str(output_dir)],
                environment=environment,
                stdin=io.StringIO(),
                stdout=io.StringIO(),
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                (session_dir / "evaluation.md").read_text(encoding="utf-8"),
                "# Language\nCEFR: B2\n\n# Content\nStrongest answer: the incident example.\n\n# Overall\nPractice articles and filler words.\n",
            )
            self.assertEqual(len(environment.llm.evaluation_calls), 1)
            prompt = environment.llm.evaluation_calls[0]["prompt"]
            self.assertIn("Seed Instruction:\nInterview me for a backend role", prompt)
            self.assertIn("speech recognition", prompt)
            self.assertIn("STT artifacts", prompt)
            self.assertIn("Interviewer 1: Tell me about yourself.", prompt)
            self.assertIn("Speaker 1: I work with Python and distributed systems.", prompt)
            self.assertIn("Interviewer 2: Describe a production incident.", prompt)
            self.assertIn("Speaker 2: I traced a queue backlog to a bad retry policy.", prompt)

    def test_evaluate_by_session_id_works_for_unclean_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            session_dir = self._write_session_fixture(
                output_dir,
                ended_cleanly=False,
                seed_instruction="Backend interview",
                turns=[
                    ("interviewer", 1, "What did you deploy recently?"),
                    ("speaker", 1, "A queue consumer with retries."),
                ],
            )
            environment = Environment.build_fake(
                llm_responses=[],
                transcripts=[],
                evaluation_responses=["# Language\n...\n\n# Content\n...\n\n# Overall\n..."],
            )

            exit_code = main(
                ["evaluate", "session-001", "--output-dir", str(output_dir)],
                environment=environment,
                stdin=io.StringIO(),
                stdout=io.StringIO(),
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((session_dir / "evaluation.md").exists())
            self.assertEqual(len(environment.llm.evaluation_calls), 1)

    def test_start_quit_writes_evaluation_after_marking_session_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            exit_code = main(
                ["start", "--seed", "Seed", "--output-dir", str(output_dir)],
                environment=Environment.build_fake(
                    llm_responses=["Opening question"],
                    transcripts=[],
                    evaluation_responses=["# Language\n...\n\n# Content\n...\n\n# Overall\n..."],
                ),
                stdin=io.StringIO("q\n"),
                stdout=io.StringIO(),
            )

            self.assertEqual(exit_code, 0)
            session_dir = self._only_session_dir(output_dir)
            metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["ended_cleanly"])
            self.assertEqual(
                (session_dir / "evaluation.md").read_text(encoding="utf-8"),
                "# Language\n...\n\n# Content\n...\n\n# Overall\n...\n",
            )

    def test_start_requires_exactly_one_seed_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with self.assertRaises(SystemExit) as missing_seed:
                main(["start", "--output-dir", str(output_dir)], stdout=io.StringIO(), stdin=io.StringIO())
            self.assertEqual(missing_seed.exception.code, 2)

            seed_file = output_dir / "seed.txt"
            seed_file.write_text("seed from file", encoding="utf-8")
            with self.assertRaises(SystemExit) as duplicate_seed:
                main(
                    [
                        "start",
                        "--seed",
                        "inline seed",
                        "--seed-file",
                        str(seed_file),
                        "--output-dir",
                        str(output_dir),
                    ],
                    stdout=io.StringIO(),
                    stdin=io.StringIO(),
                )
            self.assertEqual(duplicate_seed.exception.code, 2)

    def test_start_writes_metadata_and_keeps_aborted_session_unclean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stdout = io.StringIO()

            exit_code = main(
                ["start", "--seed", "Be strict", "--output-dir", str(output_dir)],
                environment=Environment.build_fake(
                    llm_responses=["Opening question"],
                    transcripts=[],
                ),
                stdin=io.StringIO(""),
                stdout=stdout,
            )

            self.assertEqual(exit_code, 0)
            session_dir = self._only_session_dir(output_dir)
            metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["seed_instruction"], "Be strict")
            self.assertEqual(metadata["voice"], "fake-voice")
            self.assertFalse(metadata["ended_cleanly"])
            self.assertEqual((session_dir / "interviewer_001.txt").read_text(encoding="utf-8").strip(), "Opening question")

    def test_scripted_session_persists_turns_and_reads_latest_interviewer_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stdout = io.StringIO()

            exit_code = main(
                ["start", "--seed", "Seed", "--output-dir", str(output_dir)],
                environment=Environment.build_fake(
                    llm_responses=["Opening question", "Second question"],
                    transcripts=["Accepted answer"],
                ),
                stdin=io.StringIO("r\nt\n\n\nr\nq\n"),
                stdout=stdout,
            )

            self.assertEqual(exit_code, 0)
            session_dir = self._only_session_dir(output_dir)
            self.assertEqual(
                sorted(path.name for path in session_dir.glob("*.txt")),
                ["interviewer_001.txt", "interviewer_002.txt", "speaker_001.txt"],
            )
            self.assertEqual((session_dir / "speaker_001.txt").read_text(encoding="utf-8").strip(), "Accepted answer")
            output = stdout.getvalue()
            self.assertIn("Opening question", output)
            self.assertIn("Second question", output)
            metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["ended_cleanly"])

    def test_redo_discards_take_without_persisting_speaker_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            exit_code = main(
                ["start", "--seed", "Seed", "--output-dir", str(output_dir)],
                environment=Environment.build_fake(
                    llm_responses=["Opening question"],
                    transcripts=["Discarded answer"],
                ),
                stdin=io.StringIO("t\n\nr\nq\n"),
                stdout=io.StringIO(),
            )

            self.assertEqual(exit_code, 0)
            session_dir = self._only_session_dir(output_dir)
            self.assertFalse((session_dir / "speaker_001.txt").exists())
            self.assertFalse((session_dir / "speaker_001.wav").exists())
            self.assertEqual(
                sorted(path.name for path in session_dir.glob("*.txt")),
                ["interviewer_001.txt"],
            )

    def test_seed_file_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            seed_file = output_dir / "seed.txt"
            seed_file.write_text("Seed from file", encoding="utf-8")

            exit_code = main(
                ["start", "--seed-file", str(seed_file), "--output-dir", str(output_dir)],
                environment=Environment.build_fake(
                    llm_responses=["Opening question"],
                    transcripts=[],
                ),
                stdin=io.StringIO("q\n"),
                stdout=io.StringIO(),
            )

            self.assertEqual(exit_code, 0)
            session_dir = self._only_session_dir(output_dir)
            metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["seed_instruction"], "Seed from file")
            self.assertTrue(metadata["ended_cleanly"])

    def test_start_passes_voice_to_default_environment_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            with patch("interview.cli.Environment.build_anthropic") as build_environment:
                build_environment.return_value = Environment.build_fake(
                    llm_responses=["Opening question"],
                    transcripts=[],
                    voice="nova",
                )

                exit_code = main(
                    ["start", "--seed", "Seed", "--voice", "nova", "--output-dir", str(output_dir)],
                    stdin=io.StringIO("q\n"),
                    stdout=io.StringIO(),
                )

            self.assertEqual(exit_code, 0)
            build_environment.assert_called_once_with(voice="nova")
            session_dir = self._only_session_dir(output_dir)
            metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["voice"], "nova")

    def test_start_uses_default_voice_when_flag_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            with patch("interview.cli.Environment.build_anthropic") as build_environment:
                build_environment.return_value = Environment.build_fake(
                    llm_responses=["Opening question"],
                    transcripts=[],
                    voice="alloy",
                )

                exit_code = main(
                    ["start", "--seed", "Seed", "--output-dir", str(output_dir)],
                    stdin=io.StringIO("q\n"),
                    stdout=io.StringIO(),
                )

            self.assertEqual(exit_code, 0)
            build_environment.assert_called_once_with(voice="alloy")
            session_dir = self._only_session_dir(output_dir)
            metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["voice"], "alloy")

    def test_start_requires_anthropic_api_key_when_using_default_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                    main(
                        ["start", "--seed", "Seed", "--output-dir", str(output_dir)],
                        stdin=io.StringIO("q\n"),
                        stdout=io.StringIO(),
                    )

            self.assertEqual(list(output_dir.iterdir()), [])

    def test_start_requires_openai_api_key_when_using_default_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                    main(
                        ["start", "--seed", "Seed", "--output-dir", str(output_dir)],
                        stdin=io.StringIO("q\n"),
                        stdout=io.StringIO(),
                    )

            self.assertEqual(list(output_dir.iterdir()), [])

    def _only_session_dir(self, output_dir: Path) -> Path:
        session_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
        self.assertEqual(len(session_dirs), 1)
        return session_dirs[0]

    def _write_session_fixture(
        self,
        output_dir: Path,
        *,
        ended_cleanly: bool,
        seed_instruction: str,
        turns: list[tuple[str, int, str]],
    ) -> Path:
        session_dir = output_dir / "session-001"
        session_dir.mkdir()
        (session_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": "session-001",
                    "created_at": "2026-07-01T12:00:00+00:00",
                    "seed_instruction": seed_instruction,
                    "voice": "fake-voice",
                    "providers": {
                        "llm": "fake-llm",
                        "tts": "fake-tts",
                        "stt": "fake-stt",
                        "recorder": "fake-recorder",
                        "player": "fake-player",
                    },
                    "ended_cleanly": ended_cleanly,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for speaker, turn_number, text in turns:
            (session_dir / f"{speaker}_{turn_number:03d}.txt").write_text(text + "\n", encoding="utf-8")
        return session_dir


if __name__ == "__main__":
    unittest.main()
