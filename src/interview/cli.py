from __future__ import annotations

import argparse
import sys
from pathlib import Path

from interview.environment import Environment
from interview.session_runner import SessionRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interview")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--seed")
    start_parser.add_argument("--seed-file", type=Path)
    start_parser.add_argument("--voice", default="alloy")
    start_parser.add_argument("--output-dir", type=Path, default=Path.cwd())

    return parser


def main(argv: list[str] | None = None, *, environment: Environment | None = None, stdin=None, stdout=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
