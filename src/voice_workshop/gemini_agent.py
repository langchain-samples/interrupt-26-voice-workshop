"""Raw Gemini Live 3.1 voice-agent workshop starter."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from contextlib import AsyncExitStack
from typing import Any

from google import genai
from google.genai import types
from langsmith.integrations.gemini_live import wrap_gemini_live

from .audio import MicStream, SpeakerStream, resample_pcm16
from .tracing import configure_langsmith, load_environment, require_environment

MODEL = "gemini-3.1-flash-live-preview"
VOICE = "Aoede"
MIC_SAMPLE_RATE = 24_000
SEND_SAMPLE_RATE = 16_000
RECEIVE_SAMPLE_RATE = 24_000
MAX_TRACE_AUDIO_SECONDS = 10 * 60

SYSTEM_PROMPT = """You are a friendly voice assistant. Keep every reply brief,
conversational, and easy to understand when spoken aloud. Do not use markdown,
emoji, bullet points, or other visual formatting."""


def _server_content(message: Any) -> Any:
    return getattr(message, "server_content", None)


def _audio_chunks(message: Any) -> list[bytes]:
    content = _server_content(message)
    turn = getattr(content, "model_turn", None) if content is not None else None
    chunks: list[bytes] = []
    for part in getattr(turn, "parts", None) or []:
        inline_data = getattr(part, "inline_data", None)
        data = getattr(inline_data, "data", None) if inline_data is not None else None
        if isinstance(data, bytes) and data:
            chunks.append(data)
    return chunks


async def _wait_for_workers(*tasks: asyncio.Task[None]) -> None:
    """Wait for workers, consuming only cancellation of this parent task."""
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is None or current.cancelling() == 0:
            raise
        current.uncancel()


async def run() -> None:
    """Run a local Gemini Live conversation until interrupted."""
    thread_id = str(uuid.uuid4())
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    mic = MicStream(sample_rate=MIC_SAMPLE_RATE)
    speaker = SpeakerStream(sample_rate=RECEIVE_SAMPLE_RATE)

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=SYSTEM_PROMPT,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=(
                    types.StartSensitivity.START_SENSITIVITY_HIGH
                ),
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=200,
                silence_duration_ms=800,
            )
        ),
    )

    # TODO: Uncomment these lines to configure the LangSmith project.
    # project_name = configure_langsmith("gemini")
    # print(f"LangSmith project: {project_name}")

    print(f"Gemini Live model: {MODEL}")
    print(f"Conversation: {thread_id}")

    mic_task: asyncio.Task[None] | None = None
    receive_task: asyncio.Task[None] | None = None
    try:
        async with AsyncExitStack() as stack:
            session = await stack.enter_async_context(
                client.aio.live.connect(model=MODEL, config=config)
            )

            # TODO: Uncomment this block to trace the live session.
            # session = await stack.enter_async_context(
            #     wrap_gemini_live(
            #         session,
            #         model=MODEL,
            #         thread_id=thread_id,
            #         sample_rate=RECEIVE_SAMPLE_RATE,
            #         project_name=project_name,
            #         tags=["voice-workshop", "gemini-live"],
            #         metadata={"backend": "gemini-live"},
            #         is_agent_speaking=lambda: speaker.buffered_bytes() > 0,
            #         max_audio_seconds=MAX_TRACE_AUDIO_SECONDS,
            #     )
            # )

            # TODO: Uncomment the next line to record played audio in the trace.
            # speaker.set_played_callback(session.record_agent_audio)
            mic.start()
            speaker.start()
            print("Connected. Speak into your microphone; press Control-C to stop.")

            async def pump_microphone() -> None:
                async for frame in mic.frames():
                    send_frame = resample_pcm16(
                        frame,
                        MIC_SAMPLE_RATE,
                        SEND_SAMPLE_RATE,
                    )
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=send_frame,
                            mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                        )
                    )
                    # TODO: Uncomment the next line to record user audio in the trace.
                    # session.record_user_audio(frame)

            async def pump_responses() -> None:
                while True:
                    async for message in session.receive():
                        content = _server_content(message)
                        interrupted = bool(
                            content is not None
                            and getattr(content, "interrupted", False)
                        )
                        if interrupted:
                            speaker.clear()

                        for chunk in _audio_chunks(message):
                            speaker.write(chunk)

            mic_task = asyncio.create_task(pump_microphone())
            receive_task = asyncio.create_task(pump_responses())
            await _wait_for_workers(mic_task, receive_task)
    finally:
        for task in (mic_task, receive_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (mic_task, receive_task) if task is not None),
            return_exceptions=True,
        )
        speaker.set_played_callback(None)
        mic.stop()
        speaker.stop()
        await client.aio.aclose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemini-demo",
        description="Run the Gemini Live 3.1 voice agent.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logs.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("langsmith.integrations").setLevel(logging.DEBUG)

    load_environment()
    try:
        require_environment("GOOGLE_API_KEY")
        asyncio.run(run())
    except (RuntimeError, ValueError) as exc:
        print(f"gemini-demo: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
