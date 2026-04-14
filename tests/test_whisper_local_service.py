from __future__ import annotations

from io import BytesIO

import pytest
from click.testing import CliRunner
from werkzeug.datastructures import FileStorage

from astrbot.cli.__main__ import cli
from astrbot.core.provider.whisper_local_service import (
    WhisperLocalServiceConfig,
    WhisperLocalServiceRuntime,
    create_whisper_local_service_app,
)


def _make_config() -> WhisperLocalServiceConfig:
    return WhisperLocalServiceConfig(
        host="127.0.0.1",
        port=8001,
        model_name="whisper-1",
        model_path="tiny",
        device="cpu",
        api_key="",
    )


def test_cli_registers_whisper_service_group():
    result = CliRunner().invoke(cli, ["help", "whisper-service"])

    assert result.exit_code == 0
    assert "Run the official local Whisper sidecar service." in result.output
    assert "run" in result.output


@pytest.mark.asyncio
async def test_translations_endpoint_uses_translate_task(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    async def fake_initialize(self) -> None:
        self.resolved_device = self.config.device

    async def fake_transcribe(
        self,
        file_path,
        *,
        task="transcribe",
        language=None,
        prompt=None,
        temperature=None,
    ) -> str:
        captured["task"] = task
        captured["language"] = language
        captured["prompt"] = prompt
        captured["temperature"] = temperature
        captured["suffix"] = file_path.suffix
        return "translated text"

    monkeypatch.setattr(WhisperLocalServiceRuntime, "initialize", fake_initialize)
    monkeypatch.setattr(WhisperLocalServiceRuntime, "transcribe", fake_transcribe)

    app = create_whisper_local_service_app(_make_config())

    async with app.test_app():
        test_client = app.test_client()
        response = await test_client.post(
            "/v1/audio/translations",
            files={
                "file": FileStorage(
                    stream=BytesIO(b"fake-audio"),
                    filename="speech.wav",
                    content_type="audio/wav",
                )
            },
            form={
                "model": "whisper-1",
                "language": "zh",
                "prompt": "translate this",
                "temperature": "0.25",
            },
        )

    assert response.status_code == 200
    assert await response.get_json() == {"text": "translated text"}
    assert captured["task"] == "translate"
    assert captured["language"] == "zh"
    assert captured["prompt"] == "translate this"
    assert captured["temperature"] == 0.25
    assert captured["suffix"] == ".wav"
