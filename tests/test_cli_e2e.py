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


if __name__ == "__main__":
    unittest.main()
