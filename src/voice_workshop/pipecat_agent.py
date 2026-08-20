"""OpenAI STT → LLM → TTS Pipecat workshop starter."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from langsmith.integrations.pipecat import configure_pipecat, set_thread_id
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

from .tracing import configure_langsmith, load_environment, require_environment

STT_MODEL = "gpt-4o-transcribe"
LLM_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "coral"
AUDIO_BUFFER_SIZE = 32_000
TRACE_AUDIO_SECONDS = 10 * 60
TRACE_AUDIO_BYTES = TRACE_AUDIO_SECONDS * 24_000 * 2 * 2 + 44

SYSTEM_PROMPT = """You are a friendly voice assistant. Keep every reply brief,
conversational, and easy to understand when spoken aloud. Do not use markdown,
emoji, bullet points, or other visual formatting."""
GREETING = "Hello! I'm your Pipecat voice assistant. How can I help?"


async def run() -> None:
    """Run the local Pipecat cascade until interrupted."""
    api_key = os.environ["OPENAI_API_KEY"]

    # TODO: Uncomment this block to enable LangSmith tracing.
    # project_name = configure_langsmith("pipecat")
    # conversation_id = str(uuid.uuid4())
    # set_thread_id(conversation_id)
    # span_processor = configure_pipecat(
    #     project=project_name,
    #     service_name="voice-workshop-pipecat",
    #     audio_size_limit_bytes=TRACE_AUDIO_BYTES,
    # )
    # if span_processor is None:
    #     raise RuntimeError("LangSmith Pipecat tracing could not be configured")
    # logger.info(f"LangSmith project: {project_name}")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    stt = OpenAISTTService(
        api_key=api_key,
        settings=OpenAISTTService.Settings(model=STT_MODEL),
    )
    llm = OpenAILLMService(
        api_key=api_key,
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL,
            temperature=0.3,
        ),
    )
    tts = OpenAITTSService(
        api_key=api_key,
        settings=OpenAITTSService.Settings(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            instructions="Speak naturally, clearly, and at a conversational pace.",
        ),
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )
    audio_buffer = AudioBufferProcessor(
        num_channels=2,
        buffer_size=AUDIO_BUFFER_SIZE,
    )
    # TODO: Uncomment the next line to attach conversation audio to the trace.
    # span_processor.attach_audio_buffer(audio_buffer, conversation_id)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            audio_buffer,
            context_aggregator.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=24_000,
            enable_metrics=True,
        ),
        # TODO: Uncomment these options to emit Pipecat tracing spans.
        # enable_tracing=True,
        # enable_turn_tracking=True,
        # conversation_id=conversation_id,
    )

    await audio_buffer.start_recording()
    await worker.queue_frames([TTSSpeakFrame(text=GREETING, append_to_context=True)])
    logger.info("Connected. Speak into your microphone; press Control-C to stop.")
    runner = WorkerRunner(handle_sigint=sys.platform != "win32")
    await runner.add_workers(worker)
    await runner.run()


def main() -> None:
    load_environment()
    try:
        require_environment("OPENAI_API_KEY")
        asyncio.run(run())
    except (RuntimeError, ValueError) as exc:
        print(f"pipecat-demo: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
