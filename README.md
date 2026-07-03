# raulf-the-recruiter

A voice-based CLI for practicing interviews and spoken English. An LLM plays
the Interviewer: its replies are spoken aloud via TTS, you answer by talking
into the mic, and your speech is transcribed and fed back into the
conversation. Every session is persisted to disk so it can be reviewed (and,
eventually, evaluated).

The interview is always conducted in English, regardless of the language of
the instruction that seeds it.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- A working microphone and audio output (audio I/O uses `sounddevice`)
- API keys, exported as environment variables:
  - `ANTHROPIC_API_KEY` — Interviewer turns (Claude Opus 4.8)
  - `OPENAI_API_KEY` — TTS (`gpt-4o-mini-tts`) and STT (`gpt-4o-transcribe`)

## Usage

Start a session with a Seed Instruction — the scenario, role, or question set
for the interview:

```
uv run interview start --seed "Senior backend engineer interview, focus on system design"
```

or load it from a file:

```
uv run interview start --seed-file ./seeds/backend.md
```

Options:

- `--voice NAME` — TTS voice (default: `alloy`)
- `--output-dir DIR` — where session directories are created (default: current directory)

The opening question is generated, spoken aloud, and saved. Then a menu loop
runs until you quit:

| Key | Action |
| --- | --- |
| `r` | Print the latest Interviewer text |
| `a` | Replay the latest Interviewer audio |
| `t` | Record an answer (Enter stops recording, 6-minute cap) |
| `q` | End the session |

After recording, the transcript is shown with a confirmation gate: press
Enter to accept and send it to the Interviewer, or `r` to discard the take
and re-record. Only accepted takes are persisted as turns.

## Session output

Each session gets its own directory named by a unique ID:

```
<output-dir>/<session-id>/
  metadata.json        # seed instruction, voice, providers, ended_cleanly
  interviewer_001.txt  # Interviewer turn text
  interviewer_001.wav  # its TTS audio
  speaker_001.txt      # accepted transcript
  speaker_001.wav      # your recorded audio
  ...
```

Speaker audio is kept alongside transcripts so sessions can be re-listened
to or re-transcribed later.

## Architecture

Providers (LLM, TTS, STT, recorder, player) sit behind protocols in
`src/interview/environment.py`, with fake implementations for testing and
real ones wired up in `Environment.build_anthropic`. This keeps the door
open for local backends (e.g. Whisper for STT, Piper for TTS) later. Audio
is WAV/PCM end to end; `sounddevice` + `soundfile` are the only audio
dependencies.

The full spec lives in [`specs/cli.md`](specs/cli.md) and the domain
vocabulary in [`CONTEXT.md`](CONTEXT.md).

## Development

```
uv run python -m unittest discover tests
```

## Roadmap

- End-of-session Evaluation (English level + answer quality) and an
  `interview evaluate` subcommand.
