# Whisper Local Service

如果你需要同时使用：

- 本地 `Whisper`
- AstrBot 知识库

推荐使用 **`Whisper(Local Service)` + 独立本地服务进程**，而不是同进程的 `Whisper(Local)`。

## 为什么推荐本地服务模式

`Whisper(Local)` 会在 AstrBot 主进程内加载 `whisper` / `torch`。知识库检索在实际使用时会按需加载 `faiss`。在 macOS 上，这两套原生依赖可能在同一进程内触发运行时冲突，例如：

```text
OMP Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

将本地 Whisper 放到独立进程后：

- AstrBot 主进程只保留消息平台、LLM、知识库和插件
- Whisper / Torch 在独立服务进程中运行
- 两边不再共享同一个原生运行时地址空间

## 启动 Whisper Local Service

先准备依赖：

```bash
cd /path/to/AstrBot
uv sync
uv run python -m pip install openai-whisper
```

并确保系统已安装 `ffmpeg`。

然后启动官方本地服务：

```bash
astrbot whisper-service run --model-path tiny --device mps
```

常用参数：

- `--model-name`：对外暴露的 API 模型名，默认 `whisper-1`
- `--model-path`：传给 `whisper.load_model()` 的模型名或本地路径，默认 `tiny`
- `--device`：`auto` / `cpu` / `mps` / `cuda`
- `--api-key`：可选，本地 API 的 Bearer Token
- `--host` / `--port`：服务监听地址，默认 `127.0.0.1:8001`

如果你希望把本地服务和 AstrBot 的依赖完全隔离，也可以使用独立的 Python 环境运行这个命令。

## 在 AstrBot 中配置

1. 打开 WebUI -> `服务提供商`
2. 新增或启用 `Whisper(Local Service)`
3. 填写：

```text
api_base = http://127.0.0.1:8001/v1
model = whisper-1
api_key = （留空，或与你启动服务时设置的 --api-key 一致）
```

4. 打开 `配置`
5. 设置：

```text
provider_stt_settings.enable = true
provider_stt_settings.provider_id = whisper_local_service
```

这样 AstrBot 会继续走原有的 STT 流程，但不再在主进程内加载本地 Whisper。

## 验证

先检查本地服务健康状态：

```bash
curl http://127.0.0.1:8001/health
```

预期返回：

```json
{
  "status": "ok",
  "model": "whisper-1",
  "model_path": "tiny",
  "device": "mps"
}
```

然后在 AstrBot 中启用 `Whisper(Local Service)` 后，发送一条语音消息测试转写即可。

## 何时仍然可以使用 Whisper(Local)

以下场景仍然适合 `Whisper(Local)`：

- 你只需要本地语音转文本
- 你不使用知识库
- 你接受同进程模式的可选依赖和平台限制

如果你需要和知识库长期稳定共存，优先使用本地服务模式。
