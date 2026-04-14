from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

from quart import Quart, jsonify, request

LOGGER = logging.getLogger("astrbot.whisper_local_service")


@dataclass(slots=True)
class WhisperLocalServiceConfig:
    host: str
    port: int
    model_name: str
    model_path: str
    device: str
    api_key: str


class WhisperLocalServiceRuntime:
    def __init__(self, config: WhisperLocalServiceConfig) -> None:
        self.config = config
        self.model: Any | None = None
        self.resolved_device = ""

    def _resolve_device(self) -> str:
        requested = self.config.device.lower()
        if requested == "cpu":
            return "cpu"

        try:
            import torch
        except ImportError:
            if requested == "auto":
                LOGGER.warning(
                    "PyTorch is unavailable while resolving device=auto. Falling back to CPU."
                )
                return "cpu"
            raise

        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend and mps_backend.is_available():
                return "mps"
            return "cpu"

        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            LOGGER.warning(
                "CUDA was requested but is unavailable. Falling back to CPU."
            )
            return "cpu"

        if requested == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend and mps_backend.is_available():
                return "mps"
            LOGGER.warning("MPS was requested but is unavailable. Falling back to CPU.")
            return "cpu"

        LOGGER.warning(
            "Unknown device=%s was requested. Falling back to CPU.",
            self.config.device,
        )
        return "cpu"

    async def initialize(self) -> None:
        import whisper

        loop = asyncio.get_running_loop()
        self.resolved_device = self._resolve_device()
        LOGGER.info(
            "Loading Whisper model. model_path=%s device=%s",
            self.config.model_path,
            self.resolved_device,
        )
        self.model = await loop.run_in_executor(
            None,
            partial(
                whisper.load_model,
                self.config.model_path,
                device=self.resolved_device,
            ),
        )
        LOGGER.info(
            "Whisper model is ready. model_name=%s device=%s",
            self.config.model_name,
            self.resolved_device,
        )

    async def transcribe(
        self,
        file_path: Path,
        *,
        task: Literal["transcribe", "translate"] = "transcribe",
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Whisper model is not initialized.")

        options: dict[str, Any] = {}
        options["task"] = task
        if language:
            options["language"] = language
        if prompt:
            options["initial_prompt"] = prompt
        if temperature is not None:
            options["temperature"] = temperature

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(self.model.transcribe, str(file_path), **options),
        )
        return str(result["text"])


def _build_error(message: str, *, status_code: int) -> tuple[Any, int]:
    return (
        jsonify(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            }
        ),
        status_code,
    )


def _extract_bearer_token(header_value: str | None) -> str:
    if not header_value:
        return ""
    prefix = "Bearer "
    if header_value.startswith(prefix):
        return header_value[len(prefix) :].strip()
    return ""


def create_whisper_local_service_app(config: WhisperLocalServiceConfig) -> Quart:
    app = Quart(__name__)
    runtime = WhisperLocalServiceRuntime(config)

    @app.before_serving
    async def _startup() -> None:
        await runtime.initialize()

    @app.get("/health")
    @app.get("/healthz")
    async def _health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "model": config.model_name,
                "model_path": config.model_path,
                "device": runtime.resolved_device or config.device,
            }
        )

    @app.get("/v1/models")
    async def _models() -> Any:
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": config.model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "astrbot-whisper-local-service",
                    }
                ],
            }
        )

    async def _handle_transcription(
        *,
        task: Literal["transcribe", "translate"] = "transcribe",
    ) -> Any:
        if config.api_key:
            token = _extract_bearer_token(request.headers.get("Authorization"))
            if token != config.api_key:
                return _build_error("Invalid API key.", status_code=401)

        files = await request.files
        form = await request.form

        uploaded_file = files.get("file")
        if uploaded_file is None:
            return _build_error("Missing multipart field 'file'.", status_code=400)

        requested_model = str(form.get("model", config.model_name)).strip()
        if requested_model and requested_model != config.model_name:
            return _build_error(
                f"Unknown model '{requested_model}'. Expected '{config.model_name}'.",
                status_code=400,
            )

        language = str(form.get("language", "")).strip() or None
        prompt = str(form.get("prompt", "")).strip() or None

        temperature_value = str(form.get("temperature", "")).strip()
        temperature = None
        if temperature_value:
            try:
                temperature = float(temperature_value)
            except ValueError:
                return _build_error(
                    "temperature must be a valid float.", status_code=400
                )

        suffix = Path(uploaded_file.filename or "audio.wav").suffix or ".wav"
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix="astrbot-whisper-",
            suffix=suffix,
        )
        os.close(file_descriptor)
        temp_path = Path(temp_name)
        try:
            file_data = uploaded_file.read()
            if asyncio.iscoroutine(file_data):
                file_data = await file_data
            temp_path.write_bytes(file_data)

            text = await runtime.transcribe(
                temp_path,
                task=task,
                language=language,
                prompt=prompt,
                temperature=temperature,
            )
            return jsonify({"text": text})
        finally:
            temp_path.unlink(missing_ok=True)

    @app.post("/v1/audio/transcriptions")
    async def _transcriptions() -> Any:
        return await _handle_transcription(task="transcribe")

    @app.post("/v1/audio/translations")
    async def _translations() -> Any:
        return await _handle_transcription(task="translate")

    return app


def run_whisper_local_service(config: WhisperLocalServiceConfig) -> None:
    app = create_whisper_local_service_app(config)
    LOGGER.info(
        "Starting Whisper Local Service on http://%s:%s using model_name=%s model_path=%s",
        config.host,
        config.port,
        config.model_name,
        config.model_path,
    )
    app.run(host=config.host, port=config.port, debug=False)
