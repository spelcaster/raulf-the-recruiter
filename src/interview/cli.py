from __future__ import annotations

import argparse
import sys
from pathlib import Path

from interview.environment import Environment
from interview.session_evaluator import write_evaluation
from interview.session_runner import SessionRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interview")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--seed")
    start_parser.add_argument("--seed-file", type=Path)
    start_parser.add_argument("--voice", default="alloy")
    start_parser.add_argument("--output-dir", type=Path, default=Path("sessions"))

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("session_id", nargs="?")
    evaluate_parser.add_argument("--last", action="store_true")
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("sessions"))

    return parser


def main(argv: list[str] | None = None, *, environment: Environment | None = None, stdin=None, stdout=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        if bool(args.session_id) == bool(args.last):
            parser.error("exactly one of <session-id> or --last is required")
        session_dir = _resolve_session_dir(args.output_dir, session_id=args.session_id, use_last=args.last)
        llm = environment.llm if environment is not None else Environment.build_anthropic_llm()
        write_evaluation(session_dir, llm=llm)
        return 0

    if args.command != "start":
        parser.error("a subcommand is required")

    if (args.seed is None) == (args.seed_file is None):
        parser.error("exactly one of --seed or --seed-file is required")

    seed_instruction = args.seed
    if args.seed_file is not None:
        seed_instruction = args.seed_file.read_text(encoding="utf-8")

    runtime_environment = environment or Environment.build_anthropic(voice=args.voice)
    runner = SessionRunner(
        environment=runtime_environment,
        stdin=stdin or sys.stdin,
        stdout=stdout or sys.stdout,
        output_dir=args.output_dir,
    )
    return runner.start(seed_instruction=seed_instruction)

def _resolve_session_dir(output_dir: Path, *, session_id: str | None, use_last: bool) -> Path:
    if use_last:
        if not output_dir.is_dir():
            raise RuntimeError(f"no session directories found in {output_dir}")
        session_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir())
        if not session_dirs:
            raise RuntimeError(f"no session directories found in {output_dir}")
        return max(session_dirs, key=lambda path: path.stat().st_mtime)

    assert session_id is not None
    session_dir = output_dir / session_id
    if not session_dir.is_dir():
        raise RuntimeError(f"session {session_id} not found in {output_dir}")
    return session_dir


if __name__ == "__main__":
    raise SystemExit(main())
