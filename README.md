# Voice Agent Workshop — Interrupt 2026 NYC

Welcome to the voice agent workshop at Interrupt 2026 in New York City. In this
workshop, you will run two interruptible voice agents, talk to them through
your computer's microphone and speakers, and add LangSmith tracing:

- `pipecat-demo` is an OpenAI voice pipeline built with Pipecat:
  `gpt-4o-transcribe` → `gpt-4o-mini` → `gpt-4o-mini-tts`.
- `gemini-demo` is a native speech-to-speech agent built with Gemini Live 3.1.

Both starter agents support interruption and run locally without LangSmith.
During the workshop, you will enable the supplied tracing code and inspect what
happens at each stage of a live voice interaction.

> Audio is always sent to the selected model provider. After you enable tracing,
> audio and transcripts are also sent to LangSmith. Use only non-sensitive
> workshop conversations. Trace audio is limited to the first ten minutes.


## Requirements

- Python 3.11+ (the checked-in `.python-version` selects Python 3.12 for uv)
- A working microphone and speaker
- PortAudio (on macOS: `brew install portaudio`)
- OpenAI and Google API keys
- A LangSmith API key for the tracing exercise

## Setup with uv

```bash
uv sync
cp .env.example .env
```

Fill in `OPENAI_API_KEY` and `GOOGLE_API_KEY` in `.env`. The file is ignored by
Git. You will add `LANGSMITH_API_KEY` during the tracing exercise.

Run either agent:

```bash
uv run pipecat-demo
uv run gemini-demo
```

Press Control-C to stop.


## Setup with pip

The same console commands work without uv:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
pipecat-demo
gemini-demo
```

## Workshop exercise: add LangSmith tracing

Let's add tracing!

1. Add `LANGSMITH_API_KEY` to `.env`.
2. Open `src/voice_workshop/pipecat_agent.py` or
   `src/voice_workshop/gemini_agent.py`.
3. Find each `TODO` and uncomment its supplied tracing code.
4. Run the demo again and inspect the resulting trace in LangSmith.


```bash
uv run pipecat-demo
```

## What to inspect in LangSmith

The Pipecat trace shows the full `STT → LLM → TTS` cascade beneath each turn.
The Gemini trace shows the provider's live speech-to-speech events. Both attach
a bounded stereo WAV with the user and agent on separate channels. Interrupt
the agent while it is speaking and compare the resulting turn and audio.

By default, traces go to `interrupt-voice-workshop-pipecat` and
`interrupt-voice-workshop-gemini`.

## Tests

The tests use synthetic PCM data and do not access the microphone or network:

```bash
uv run python -m unittest discover -s tests -v
```
