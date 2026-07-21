#!/bin/bash
# ============================================================
#  AudiobookStudio — бутстрап инстанса Vast.ai (идемпотентный)
#  Базовый образ: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
#  (python3/pip/torch уже есть). Лог: /workspace/setup.log
#  Запускается деплоем: nohup bash setup_server.sh
# ============================================================
set -u

# Весь вывод — в лог установки
exec >> /workspace/setup.log 2>&1
echo "=== setup_server.sh: старт $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="

cd /workspace

# --- env контейнера ---
# Скрипт запускается через ssh-сессию, которая НЕ наследует переменные
# окружения docker-контейнера (-e при создании инстанса). Подхватываем их
# у PID 1 (главного процесса контейнера).
while IFS= read -r -d '' kv; do
    case "$kv" in
        AUTH_TOKEN=*|MODEL_ID=*|VOICE_SAMPLE=*|PORT=*|REF_WINDOW_MODE=*|TTS_ENGINE=*|TTS_XVEC_ONLY=*|TTS_TEMPERATURE=*|VLLM_PORT=*|VLLM_CONCURRENCY=*|VLLM_START_TIMEOUT_S=*|VLLM_READ_TIMEOUT_S=*) export "$kv";;
        # фичи v4 (по умолчанию выкл) + ключ/ид для сторожа-«мертвеца»
        STRESS_MARKING=*|ASR_QC=*|ASR_QC_CER=*|ASR_QC_RETRIES=*|DEADMAN=*|DEADMAN_MAX_HOURS=*|DEADMAN_IDLE_MIN=*|CONTAINER_API_KEY=*|CONTAINER_ID=*|VAST_CONTAINERLABEL=*) export "$kv";;
    esac
done < /proc/1/environ
if [ -z "${AUTH_TOKEN:-}" ]; then
    echo "ОШИБКА: AUTH_TOKEN не найден ни в env, ни в /proc/1/environ"
    exit 1
fi
echo "env контейнера подхвачен (AUTH_TOKEN установлен, MODEL_ID=${MODEL_ID:-по умолчанию})"

# Идемпотентность: сервер уже работает — второй раз ничего не делаем
if pgrep -f "python3 server.py" > /dev/null 2>&1; then
    echo "server.py уже запущен (pid: $(pgrep -f 'python3 server.py' | tr '\n' ' ')) — выходим"
    exit 0
fi

# --- ffmpeg (нужен серверу для перекодировки сэмпла голоса) ---
if command -v ffmpeg > /dev/null 2>&1; then
    echo "ffmpeg уже установлен: $(command -v ffmpeg)"
else
    echo "Ставлю ffmpeg через apt-get..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq ffmpeg
fi
# gcc+g++ нужны vLLM (inductor/triton/nvcc компилируют ядра на лету)
if ! command -v g++ > /dev/null 2>&1; then
    echo "Ставлю gcc и g++ (нужны vLLM для компиляции ядер)..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get install -y -qq gcc g++
fi

# --- определяем тип образа ---
# Официальный образ vllm/vllm-omni: vllm уже в PATH, стек согласован (torch,
# flashinfer, transformers 5). На нём qwen-tts НЕ ставим (утащит transformers 4
# и сломает vllm) — движок только vllm, без inprocess-фолбэка.
OFFICIAL_VLLM=0
if command -v vllm > /dev/null 2>&1; then
    OFFICIAL_VLLM=1
    echo "Обнаружен предустановленный vllm ($(command -v vllm)) — официальный образ"
fi

# --- python-зависимости ---
# PyPI с хостов Vast бывает недоступен с первого раза — ретраи обязательны.
# httpx нужен серверу для HTTP-запросов к локальному vLLM (движок v3).
# num2words — для ASR-QC (цифры транскрипта -> слова перед сравнением CER)
if [ "$OFFICIAL_VLLM" = "1" ]; then
    PKGS="fastapi uvicorn httpx faster-whisper hf_transfer num2words"
else
    PKGS="qwen-tts fastapi uvicorn faster-whisper hf_transfer httpx num2words"
fi
echo "Ставлю python-зависимости (pip, до 5 попыток): $PKGS"
PIP_OK=0
for attempt in 1 2 3 4 5; do
    if pip install -q --timeout 120 --retries 10 $PKGS; then
        PIP_OK=1
        break
    fi
    echo "pip: попытка $attempt не удалась, повтор через 20 с..."
    sleep 20
done
if [ "$PIP_OK" != "1" ]; then
    echo "ОШИБКА: pip не смог установить зависимости за 5 попыток (сеть хоста?)"
    exit 1
fi

# --- разметка ударений (STRESS_MARKING=1; провал НЕ фатален) ---
# silero-stress ставим только когда фича включена: сервер при отсутствии пакета
# сам отключит авторазметку (WARNING), словарь ударений продолжит работать.
if [ "${STRESS_MARKING:-0}" = "1" ]; then
    echo "STRESS_MARKING=1 — ставлю silero-stress (не фатально)..."
    pip install -q --timeout 120 --retries 10 silero-stress \
        || echo "WARNING: silero-stress не установился — сервер отключит авторазметку сам"
fi

# --- vLLM в ОТДЕЛЬНОМ venv (основной движок v3; провал НЕ фатален) ---
# Тупик версий: vllm 0.24+ требует transformers>=5, а qwen_tts (движок-фолбэк)
# живёт только с transformers==4.57.3. Решение — отдельный venv для vLLM
# с --system-site-packages (torch переиспользуется из системы, transformers 5
# ставится В venv и перекрывает системный 4.x только для vLLM).
VLLM_VENV=/opt/vllm-venv
if [ "${TTS_ENGINE:-auto}" = "inprocess" ]; then
    echo "TTS_ENGINE=inprocess — установка vLLM пропущена (экономия ~10 мин)"
elif [ "$OFFICIAL_VLLM" = "1" ]; then
    export VLLM_BIN="$(command -v vllm)"
    export TTS_ENGINE="${TTS_ENGINE:-vllm}"   # на официальном образе фолбэка нет
    echo "vllm официального образа: $VLLM_BIN (движок: $TTS_ENGINE)"
elif [ -x "$VLLM_VENV/bin/vllm" ]; then
    echo "vllm-venv уже готов: $VLLM_VENV/bin/vllm"
else
    echo "Создаю venv для vLLM и ставлю vllm+vllm-omni (до 5 попыток; провал не фатален)..."
    python3 -m venv --system-site-packages "$VLLM_VENV"
    VLLM_PIP_OK=0
    for attempt in 1 2 3 4 5; do
        if "$VLLM_VENV/bin/pip" install -q --timeout 120 --retries 10 \
                "vllm==0.24.*" "vllm-omni==0.24.0" "transformers>=5"; then
            VLLM_PIP_OK=1
            break
        fi
        echo "pip(vllm-venv): попытка $attempt не удалась, повтор через 20 с..."
        sleep 20
    done
    if [ "$VLLM_PIP_OK" = "1" ] && [ -x "$VLLM_VENV/bin/vllm" ]; then
        echo "vllm-venv готов ($("$VLLM_VENV/bin/pip" show transformers 2>/dev/null | grep Version))"
    else
        echo "WARNING: vllm-venv не собрался — сервер запустится, движок откатится на inprocess"
    fi
    export VLLM_BIN="$VLLM_VENV/bin/vllm"
fi

# Контроль: без этих импортов сервер бессмысленно запускать (vllm НЕ обязателен)
if [ "$OFFICIAL_VLLM" = "1" ]; then
    IMPORTS="import fastapi, uvicorn, faster_whisper"
else
    IMPORTS="import fastapi, uvicorn, qwen_tts, faster_whisper"
fi
if ! python3 -c "$IMPORTS" 2>> /workspace/setup.log; then
    echo "ОШИБКА: зависимости установлены не полностью (импорт не прошёл)"
    exit 1
fi
echo "Зависимости установлены и импортируются"

# --- запуск сервера в фоне ---
if [ ! -f /workspace/server.py ]; then
    echo "ОШИБКА: /workspace/server.py не найден — деплой не докинул файлы"
    exit 1
fi
echo "Запускаю server.py (лог: /workspace/server.log)..."
# Многопоточная загрузка модели с HuggingFace (в разы быстрее на слабых каналах)
export HF_HUB_ENABLE_HF_TRANSFER=1
nohup python3 server.py > /workspace/server.log 2>&1 &
echo "server.py запущен, pid=$!"
echo "=== setup_server.sh: готово $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
