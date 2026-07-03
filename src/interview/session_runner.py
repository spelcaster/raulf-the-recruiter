from __future__ import annotations

import io
import json
import select
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from interview.environment import Environment
from interview.session_evaluator import write_evaluation
from interview.session_metadata import SessionMetadata


MAX_RECORDING_SECONDS = 6 * 60


class SessionRunner:
    def __init__(
        self,
        *,
        environment: Environment,
        stdout,
        stdin,
        output_dir: Path,
        interviewer_name: str = "Raulf",
    ) -> None:
        self._environment = environment
        self._stdout = stdout
        self._stdin = stdin
        self._output_dir = output_dir
        self._interviewer_name = interviewer_name

    def start(self, *, seed_instruction: str) -> int:
        session_id = uuid4().hex
        session_dir = self._output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        self._status(f"Session {session_id} started. Files will be saved to {session_dir}.")

        metadata = SessionMetadata(
            id=session_id,
            created_at=datetime.now(UTC).isoformat(),
            seed_instruction=seed_instruction,
            interviewer_name=self._interviewer_name,
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

        self._status(f"{self._interviewer_name} is preparing the opening question...")
        opening = self._environment.llm.next_turn(
            seed_instruction=seed_instruction,
            history=self._read_turn_history(session_dir),
        )
        self._write_interviewer_turn(session_dir, 1, opening)

        interviewer_turn = 1
        speaker_turn = 0

        self._print_menu()
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
                self._status("Replaying the latest question audio...")
                self._environment.player.play(self._read_latest_interviewer_audio(session_dir))
                continue

            if normalized == "t":
                self._status("Recording... Press Enter to stop.")
                recording = self._environment.recorder.record(
                    stop_requested=self._stop_requested,
                    max_duration_seconds=MAX_RECORDING_SECONDS,
                )
                self._status("Transcribing your answer...")
                transcript = self._environment.stt.transcribe(recording)
                self._status(f"Transcript: {transcript}")
                decision = self._prompt("Press Enter to accept or r to redo")
                if decision is None:
                    return 0
                if decision.strip().lower() == "r":
                    continue

                speaker_turn += 1
                self._write_speaker_turn(session_dir, speaker_turn, transcript, recording)
                interviewer_turn += 1
                self._status(f"{self._interviewer_name} is preparing the next question...")
                next_turn = self._environment.llm.next_turn(
                    seed_instruction=seed_instruction,
                    history=self._read_turn_history(session_dir),
                )
                self._write_interviewer_turn(session_dir, interviewer_turn, next_turn)
                self._print_menu()
                continue

            if normalized == "q":
                metadata.ended_cleanly = True
                self._write_metadata(session_dir, metadata)
                self._status("Evaluating the session... This may take a moment.")
                evaluation_path = write_evaluation(session_dir, llm=self._environment.llm)
                self._status(f"Evaluation saved to {evaluation_path}.")
                return 0

            self._status("Unknown command")
            self._print_menu()

    def _prompt(self, label: str) -> str | None:
        self._stdout.write(f"{label}: ")
        self._flush()
        line = self._stdin.readline()
        if line == "":
            return None
        return line.rstrip("\n")

    def _status(self, message: str) -> None:
        self._stdout.write(f"{message}\n")
        self._flush()

    def _print_menu(self) -> None:
        self._stdout.write(
            "\n"
            "Commands:\n"
            "  r  show the latest question as text\n"
            "  t  record your answer (Enter stops the recording)\n"
            "  a  replay the latest question audio\n"
            "  q  end the session and generate the evaluation\n"
        )
        self._flush()

    def _flush(self) -> None:
        flush = getattr(self._stdout, "flush", None)
        if flush is not None:
            flush()

    def _write_metadata(self, session_dir: Path, metadata: SessionMetadata) -> None:
        (session_dir / "metadata.json").write_text(
            json.dumps(metadata.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_turn(self, session_dir: Path, speaker: str, turn_number: int, text: str) -> None:
        filename = f"{speaker}_{turn_number:03d}.txt"
        (session_dir / filename).write_text(text + "\n", encoding="utf-8")

    def _write_interviewer_turn(self, session_dir: Path, turn_number: int, text: str) -> None:
        self._write_turn(session_dir, "interviewer", turn_number, text)
        self._status(f"\n{self._interviewer_name}: {text}\n")
        self._status("Synthesizing the question audio...")
        audio = self._environment.tts.synthesize(text)
        (session_dir / f"interviewer_{turn_number:03d}.wav").write_bytes(audio)
        self._status("Playing the question...")
        self._environment.player.play(audio)

    def _write_speaker_turn(self, session_dir: Path, turn_number: int, text: str, audio: bytes) -> None:
        self._write_turn(session_dir, "speaker", turn_number, text)
        (session_dir / f"speaker_{turn_number:03d}.wav").write_bytes(audio)

    def _read_latest_interviewer_turn(self, session_dir: Path) -> str:
        interviewer_files = sorted(session_dir.glob("interviewer_*.txt"))
        return interviewer_files[-1].read_text(encoding="utf-8").strip()

    def _read_latest_interviewer_audio(self, session_dir: Path) -> bytes:
        interviewer_files = sorted(session_dir.glob("interviewer_*.wav"))
        return interviewer_files[-1].read_bytes()

    def _read_turn_history(self, session_dir: Path) -> list[str]:
        # Order by turn number, interviewer before speaker within the same
        # number — a plain lexical sort would group all interviewer files first.
        turn_files = sorted(session_dir.glob("*.txt"), key=self._turn_order)
        return [path.read_text(encoding="utf-8").strip() for path in turn_files]

    @staticmethod
    def _turn_order(path: Path) -> tuple[int, int]:
        speaker, number = path.stem.rsplit("_", 1)
        return (int(number), 0 if speaker == "interviewer" else 1)

    def _stop_requested(self) -> bool:
        if not self._stdin_has_data(timeout_seconds=0.05):
            return False
        self._stdin.readline()
        return True

    def _stdin_has_data(self, *, timeout_seconds: float) -> bool:
        try:
            stream = self._stdin
            fileno = stream.fileno()
        except (AttributeError, io.UnsupportedOperation, ValueError):
            fileno = None

        if fileno is not None:
            readable, _, _ = select.select([stream], [], [], timeout_seconds)
            return bool(readable)

        if hasattr(self._stdin, "tell") and hasattr(self._stdin, "seek"):
            position = self._stdin.tell()
            self._stdin.seek(0, 2)
            end = self._stdin.tell()
            self._stdin.seek(position)
            return position < end

        return False
