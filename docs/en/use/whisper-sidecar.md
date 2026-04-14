# Whisper Local Service

If you need to use both:

- local `Whisper`
- AstrBot Knowledge Base

prefer **`Whisper(Local Service)` + a separate local service process** instead of in-process `Whisper(Local)`.

## Why the local service mode is recommended

`Whisper(Local)` loads `whisper` / `torch` inside the AstrBot main process. Knowledge base retrieval loads `faiss` on demand when retrieval actually runs. On macOS, these native stacks can conflict inside the same process and fail with errors such as:

```text
OMP Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Running local Whisper in a separate process changes the boundary:

- the AstrBot main process keeps messaging platforms, LLMs, knowledge base, and plugins
- Whisper / Torch stay in the dedicated service process
- the two processes no longer share the same native runtime address space

## Start Whisper Local Service

Prepare dependencies first:

```bash
cd /path/to/AstrBot
uv sync
uv run python -m pip install openai-whisper
```

Make sure `ffmpeg` is installed on the system.

Then start the official local service:

```bash
astrbot whisper-service run --model-path tiny --device mps
```

Common options:

- `--model-name`: API-visible model name, default `whisper-1`
- `--model-path`: model name or local path passed to `whisper.load_model()`, default `tiny`
- `--device`: `auto` / `cpu` / `mps` / `cuda`
- `--api-key`: optional Bearer token for the local API
- `--host` / `--port`: bind address, default `127.0.0.1:8001`

If you want stricter dependency isolation, run the local service in a separate Python environment from AstrBot.

## Configure AstrBot

1. Open WebUI -> `Service Providers`
2. Add or enable `Whisper(Local Service)`
3. Set:

```text
api_base = http://127.0.0.1:8001/v1
model = whisper-1
api_key = (leave empty, or match the --api-key used by the local service)
```

4. Open `Configuration`
5. Set:

```text
provider_stt_settings.enable = true
provider_stt_settings.provider_id = whisper_local_service
```

AstrBot keeps the existing STT pipeline, but the main process no longer imports local Whisper.

## Verify

Check the local service health first:

```bash
curl http://127.0.0.1:8001/health
```

Expected response:

```json
{
  "status": "ok",
  "model": "whisper-1",
  "model_path": "tiny",
  "device": "mps"
}
```

Then enable `Whisper(Local Service)` in AstrBot and send a voice message to validate transcription.

## When Whisper(Local) is still fine

`Whisper(Local)` is still reasonable when:

- you only need local speech-to-text
- you are not using the knowledge base
- you accept the optional dependency and platform constraints of in-process mode

If you need stable coexistence with the knowledge base, prefer the local service pattern.
