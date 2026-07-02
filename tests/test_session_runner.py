from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interview.environment import Environment
from interview.session_runner import SessionRunner


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


if __name__ == "__main__":
    unittest.main()
