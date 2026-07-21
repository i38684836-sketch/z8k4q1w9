# AudiobookStudio — образ TTS-сервера с запечённой моделью (этап 1, план ночи 20-21.07.2026)
# Порядок слоёв: тяжёлое и стабильное — внизу, лёгкое и частое — наверху.
# Правка server.py = пересборка/перекачка только верхнего слоя (килобайты).
#
# Сборка: GitHub Actions (.github/workflows/build-image.yml) → ghcr.io (public).
# Голос НЕ запекается — voice_sample едет scp при деплое (личные данные).

# База: официальный vllm-omni (слои часто закэшированы на хостах Vast).
# Актуальные теги: hub.docker.com/r/vllm/vllm-omni/tags (v0.25.0 НЕ существует —
# проверено 21.07.2026, свежайший релизный: v0.24.0 = latest)
FROM vllm/vllm-omni:v0.24.0

# --- Слой 1: системные пакеты ---
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# --- Слой 2: python-зависимости сервера (vllm/torch уже в базе) ---
RUN pip install --no-cache-dir \
    fastapi uvicorn soundfile numpy hf_transfer \
    faster-whisper silero-stress num2words

# --- Слой 3: МОДЕЛЬ (~3.4 ГБ, меняется никогда) ---
ENV HF_HOME=/opt/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')
PY

# --- Слой 4: веса ASR-контроля качества (faster-whisper small, ~0.5 ГБ) ---
RUN python - <<'PY'
from faster_whisper import WhisperModel
WhisperModel('small', device='cpu', compute_type='int8')
PY

# --- Слой 5: inprocess-fallback (qwen-tts требует transformers==4.57.3,
#     несовместимо с vllm — отдельный venv; torch наследуем от базы) ---
RUN python -m venv --system-site-packages /opt/qwen-venv && \
    /opt/qwen-venv/bin/pip install --no-cache-dir qwen-tts "transformers==4.57.3"

# --- Слой 6: код сервера (крошечный, меняется чаще всего — ВСЕГДА последний) ---
COPY server/server.py /opt/app/server.py
COPY server/setup_server.sh /opt/app/setup_server.sh
COPY config/stress_dict.yaml /opt/app/stress_dict.yaml

ENV TTS_ENGINE=auto
ENV VLLM_BIN=vllm
ENV MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV QWEN_VENV=/opt/qwen-venv
EXPOSE 8000
# Запуск — через onstart Vast (как сейчас), CMD не задаём: Vast перекрывает entrypoint.
