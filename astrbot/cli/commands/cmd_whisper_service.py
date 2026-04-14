import logging

import click


@click.group(name="whisper-service")
def whisper_service() -> None:
    """Run the official local Whisper sidecar service."""


@whisper_service.command(name="run")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8001, show_default=True, type=int)
@click.option(
    "--model-name",
    default="whisper-1",
    envvar="ASTRBOT_WHISPER_API_MODEL",
    show_default=True,
    help="Model name exposed through the OpenAI-compatible API.",
)
@click.option(
    "--model-path",
    default="tiny",
    envvar="ASTRBOT_WHISPER_MODEL_PATH",
    show_default=True,
    help="Model name or local path passed to whisper.load_model().",
)
@click.option(
    "--device",
    default="auto",
    envvar="ASTRBOT_WHISPER_DEVICE",
    show_default=True,
    type=click.Choice(["auto", "cpu", "mps", "cuda"], case_sensitive=False),
    help="Inference device. 'auto' prefers cuda, then mps, then cpu.",
)
@click.option(
    "--api-key",
    default="",
    envvar="ASTRBOT_WHISPER_API_KEY",
    show_default=False,
    help="Optional Bearer token required by the local API.",
)
def run_whisper_service(
    host: str,
    port: int,
    model_name: str,
    model_path: str,
    device: str,
    api_key: str,
) -> None:
    """Start the local Whisper service."""
    from astrbot.core.provider.whisper_local_service import (
        WhisperLocalServiceConfig,
        run_whisper_local_service,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )
    config = WhisperLocalServiceConfig(
        host=host,
        port=port,
        model_name=model_name,
        model_path=model_path,
        device=device.lower(),
        api_key=api_key,
    )
    run_whisper_local_service(config)
