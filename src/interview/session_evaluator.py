from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from interview.llm import LLM


EVALUATION_FILENAME = "evaluation.md"


@dataclass(frozen=True)
class SessionEvaluationInput:
    seed_instruction: str
    transcript: str


def write_evaluation(session_dir: Path, *, llm: LLM) -> Path:
    evaluation_input = load_evaluation_input(session_dir)
    evaluation = llm.evaluate_session(prompt=_evaluation_prompt(evaluation_input))
    evaluation_path = session_dir / EVALUATION_FILENAME
    evaluation_path.write_text(evaluation.rstrip() + "\n", encoding="utf-8")
    return evaluation_path


def load_evaluation_input(session_dir: Path) -> SessionEvaluationInput:
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    transcript_lines = []
    for speaker, turn_number, text in _ordered_turns(session_dir):
        label = "Interviewer" if speaker == "interviewer" else "Speaker"
        transcript_lines.append(f"{label} {turn_number}: {text}")
    return SessionEvaluationInput(
        seed_instruction=metadata["seed_instruction"],
        transcript="\n".join(transcript_lines),
    )


def _ordered_turns(session_dir: Path) -> list[tuple[str, int, str]]:
    turns: list[tuple[str, int, str]] = []
    for path in session_dir.glob("*.txt"):
        if path.name == EVALUATION_FILENAME:
            continue
        prefix, turn_number = path.stem.split("_")
        if prefix not in {"interviewer", "speaker"}:
            continue
        turns.append((prefix, int(turn_number), path.read_text(encoding="utf-8").strip()))
    return sorted(turns, key=lambda turn: (turn[1], 0 if turn[0] == "interviewer" else 1))


def _evaluation_prompt(evaluation_input: SessionEvaluationInput) -> str:
    return (
        "You are evaluating an English-language mock interview after the session has ended.\n"
        "The transcript below came from speech recognition, so minor STT artifacts such as punctuation errors "
        "or homophone mistakes should not be nitpicked unless they materially affect meaning.\n"
        "Use the persisted Seed Instruction as the scenario for judging content quality.\n"
        "Return qualitative feedback only. Do not include numeric scores.\n"
        "Write markdown with these sections:\n"
        "1. Language\n"
        "- CEFR grade (A1-C2)\n"
        "- Incorrect word usage as what you said -> what a native speaker would say -> why\n"
        "- Filler-word count plus the actual fillers used\n"
        "- Recurring grammar patterns only\n"
        "2. Content\n"
        "- Short per-question assessment\n"
        "- Strongest answer\n"
        "- Weakest answer\n"
        "3. Overall\n"
        "- A 2-3 sentence summary\n"
        "- Top 3 things to practice next\n\n"
        f"Seed Instruction:\n{evaluation_input.seed_instruction.strip()}\n\n"
        f"Transcript:\n{evaluation_input.transcript}"
    )
