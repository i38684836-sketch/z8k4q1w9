# AudiobookStudio — МИНИМАЛЬНЫЙ образ TTS-сервера (только vLLM + модель).
#
# УРОК 22.07: пред. образ ставил silero-stress/faster-whisper через pip, и silero-stress
# подтянул torch под CUDA 12, затёрший «родной» torch базы (CUDA 13) → vLLM падал с
# "libcublas.so.12 not found" на КАЖДОМ хосте. Правило: в базовый python НЕ ставить
# ничего, что тянет/меняет torch. Серверные фичи (ударения silero, ASR-QC faster-whisper)
# вернём позже в ОТДЕЛЬНОМ venv, не трогая базовый torch.
#
# Что валидируем этим образом: чистый vLLM-синтез Qwen3-TTS с клоном голоса.
# Ударения работают на КЛИЕНТЕ (словарь), нормализация цифр/«Му»/чанкинг — тоже клиент.

FROM vllm/vllm-omni:v0.24.0

# ffmpeg — на всякий случай для будущей серверной постобработки
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Лёгкие зависимости сервера. НИ ОДНА не тянет torch (проверено): fastapi/uvicorn/httpx/
# soundfile/numpy/hf_transfer/faster-whisper (ctranslate2, НЕ torch). Базовый CUDA-13
# torch остаётся нетронутым. faster-whisper нужен для транскрипции референса (ref_text
# для клона) — гоняется на CPU (REF_WHISPER_DEVICE=cpu), CUDA-конфликта нет.
RUN pip install --no-cache-dir fastapi uvicorn httpx soundfile numpy hf_transfer faster-whisper

# Модель Qwen3-TTS запечена (vllm serve поднимет её локально, без скачивания на старте)
ENV HF_HOME=/opt/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')
PY

# Модель faster-whisper small для CPU-транскрипции референса (запечена, не качается на старте)
RUN python - <<'PY'
from faster_whisper import WhisperModel
WhisperModel('small', device='cpu', compute_type='int8')
PY

# Код сервера — верхним слоем (правка = перекачка килобайтов)
COPY server/server.py /opt/app/server.py
COPY server/setup_server.sh /opt/app/setup_server.sh
COPY config/stress_dict.yaml /opt/app/stress_dict.yaml

ENV TTS_ENGINE=vllm
ENV VLLM_BIN=vllm
ENV MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV REF_WHISPER_DEVICE=cpu
EXPOSE 8000
# Запуск — через onstart Vast (setup_server.sh), не через CMD.
# Голос НЕ в образе — voice_sample едет scp при деплое.
