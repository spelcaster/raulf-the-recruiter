# raulf-the-recruiter

> Friends who tried this said it was scary good. If it helped you too,
> [buy me a beer 🍺](https://ko-fi.com/hugodcarmo) or
> [sponsor me on GitHub](https://github.com/sponsors/spelcaster/) — I promise
> to drink it while fixing your GitHub issues.

A voice-based CLI for practicing interviews and spoken English. An LLM plays
the Interviewer: its replies are spoken aloud via TTS, you answer by talking
into the mic, and your speech is transcribed and fed back into the
conversation. When the session ends, an LLM evaluates the full transcript on
two dimensions: your spoken English and the quality of your answers.

The interview is always conducted in English, regardless of the language of
the instruction that seeds it.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- A working microphone and audio output (audio I/O uses `sounddevice`)
- API keys, exported as environment variables:
  - `ANTHROPIC_API_KEY` — Interviewer turns and Evaluation (Claude Opus 4.8)
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

- `--interviewer-name NAME` — what the Interviewer is called, on screen and in character (default: `Raulf`)
- `--voice NAME` — TTS voice (default: `alloy`)
- `--output-dir DIR` — where session directories are created (default: `sessions`)

The opening question is generated, spoken aloud, and saved. Then a menu loop
runs until you quit:

| Key | Action |
| --- | --- |
| `r` | Print the latest Interviewer text |
| `a` | Replay the latest Interviewer audio |
| `t` | Record an answer (Enter stops recording, 6-minute cap) |
| `q` | End the session and run the Evaluation |

After recording, the transcript is shown with a confirmation gate: press
Enter to accept and send it to the Interviewer, or `r` to discard the take
and re-record. Only accepted takes are persisted as turns.

### Evaluation

Quitting with `q` writes `evaluation.md` into the session directory: a
qualitative assessment (no numeric scores) covering your English — CEFR
grade, incorrect word usage, filler words, recurring grammar patterns — and
your content — per-question assessment, strongest and weakest answer — plus
a short summary with the top things to practice next.

A session that ended without `q` (crash, Ctrl-C) can still be evaluated
afterwards with the standalone subcommand:

```
uv run interview evaluate <session-id>
uv run interview evaluate --last
```

`--last` picks the most recently modified session directory. Use
`--output-dir` if sessions were created somewhere other than the default
`sessions` directory.

## Session output

Each session gets its own directory named by a unique ID:

```
<output-dir>/<session-id>/
  metadata.json        # seed instruction, interviewer name, voice, providers, ended_cleanly
  interviewer_001.txt  # Interviewer turn text
  interviewer_001.wav  # its TTS audio
  speaker_001.txt      # accepted transcript
  speaker_001.wav      # your recorded audio
  ...
  evaluation.md        # end-of-session Evaluation
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
