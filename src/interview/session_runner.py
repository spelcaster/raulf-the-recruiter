from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from interview.environment import Environment
from interview.session_metadata import SessionMetadata


class SessionRunner:
    def __init__(
        self,
        *,
        environment: Environment,
        stdout,
        stdin,
        output_dir: Path,
    ) -> None:
        self._environment = environment
        self._stdout = stdout
        self._stdin = stdin
        self._output_dir = output_dir

    def start(self, *, seed_instruction: str) -> int:
        session_id = uuid4().hex
        session_dir = self._output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=False)

        metadata = SessionMetadata(
            id=session_id,
            created_at=datetime.now(UTC).isoformat(),
            seed_instruction=seed_instruction,
            voice=self._environment.tts.voice,
            providers={
                "llm": self._environment.llm.name,
                "tts": self._environment.tts.name,
                "stt": self._environment.stt.name,
                "recorder": self._environment.recorder.name,
                "player": self._environment.player.name,
            },
            ended_cleanly=False,
        )
        self._write_metadata(session_dir, metadata)

        opening = self._environment.llm.next_turn(
            seed_instruction=seed_instruction,
            history=self._read_turn_history(session_dir),
        )
        self._write_turn(session_dir, "interviewer", 1, opening)

        interviewer_turn = 1
        speaker_turn = 0

        while True:
            command = self._prompt("(r)ead, (t)alk, (a)udio, (q)uit")
            if command is None:
                return 0
            normalized = command.strip().lower()

            if normalized == "r":
                latest = self._read_latest_interviewer_turn(session_dir)
                self._stdout.write(f"{latest}\n")
                continue

            if normalized == "a":
                self._environment.player.play(b"")
                continue

            if normalized == "t":
                recording = self._environment.recorder.record()
                transcript = self._environment.stt.transcribe(recording)
                self._stdout.write(f"Transcript: {transcript}\n")
                decision = self._prompt("Accept or redo? [a/r]")
                if decision is None:
                    return 0
                if decision.strip().lower() != "a":
                    continue

                speaker_turn += 1
                self._write_turn(session_dir, "speaker", speaker_turn, transcript)
                interviewer_turn += 1
                next_turn = self._environment.llm.next_turn(
                    seed_instruction=seed_instruction,
                    history=self._read_turn_history(session_dir),
                )
                self._write_turn(session_dir, "interviewer", interviewer_turn, next_turn)
                continue

            if normalized == "q":
                metadata.ended_cleanly = True
                self._write_metadata(session_dir, metadata)
                return 0

            self._stdout.write("Unknown command\n")

    def _prompt(self, label: str) -> str | None:
        self._stdout.write(f"{label}: ")
        line = self._stdin.readline()
        if line == "":
            return None
        return line.rstrip("\n")

    def _write_metadata(self, session_dir: Path, metadata: SessionMetadata) -> None:
        (session_dir / "metadata.json").write_text(
            json.dumps(metadata.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_turn(self, session_dir: Path, speaker: str, turn_number: int, text: str) -> None:
        filename = f"{speaker}_{turn_number:03d}.txt"
        (session_dir / filename).write_text(text + "\n", encoding="utf-8")

    def _read_latest_interviewer_turn(self, session_dir: Path) -> str:
        interviewer_files = sorted(session_dir.glob("interviewer_*.txt"))
        return interviewer_files[-1].read_text(encoding="utf-8").strip()

    def _read_turn_history(self, session_dir: Path) -> list[str]:
        turn_files = sorted(session_dir.glob("*.txt"))
        return [path.read_text(encoding="utf-8").strip() for path in turn_files]
