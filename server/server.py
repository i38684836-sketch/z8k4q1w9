# -*- coding: utf-8 -*-
"""AudiobookStudio — серверная часть (запускается на инстансе Vast.ai), v3.

FastAPI + Qwen3-TTS: клонирование голоса по сэмплу и озвучка чанков текста.

API (протокол заданий v2 — соединения живут секунды, злые NAT их не душат):
  GET  /health              — без авторизации: {"status","ready","model","engine",
                              "queue","jobs","error"}
  POST /tts                 — Authorization: Bearer <AUTH_TOKEN>,
                              JSON {"id","text","language"} -> МГНОВЕННЫЙ ответ
                              200 {"status":"queued"|"processing"|"done"|"error"}.
                              Идемпотентен: повтор с тем же id не создаёт дублей;
                              повтор для error/выданного done — перезапускает задание.
  GET  /result/{id}         — Authorization: Bearer <AUTH_TOKEN>;
                              404 — задание неизвестно/отменено; 202 {"status":...} — ещё
                              не готово; 500 {"error":...} — генерация упала;
                              200 WAV bytes (24000 Hz, mono, s16) + X-Chunk-Id — готово.
                              Результат хранится до вытеснения (LRU, JOBS_CAP штук).
  DELETE /job/{id}          — Authorization: Bearer <AUTH_TOKEN>; отмена устаревшего
                              задания (клиент передал чанк другому серверу после
                              дедлайна): queued -> cancelled (воркеру не отдаётся),
                              processing -> cancelled (генерацию не прервать, но
                              результат НЕ сохраняется), done/error/cancelled ->
                              200 без изменений (идемпотентно), неизвестный id ->
                              404. Повторный POST /tts с тем же id перезапускает
                              отменённое задание.

Движки (v3, внешний контракт НЕ изменён ни на байт):
  vllm       — основной: после подготовки референса стартуем сабпроцессом локальный
               vLLM-сервер (vllm serve <MODEL_ID> --omni --port VLLM_PORT, лог в
               /workspace/vllm.log), ждём готовности поллингом до VLLM_START_TIMEOUT_S,
               регистрируем голос: способ (а) POST /v1/audio/voices (именованный голос
               user_voice), при любом его сбое — способ (б) ref_audio+ref_text в каждом
               запросе. Задания уходят HTTP-запросами POST /v1/audio/speech, до
               VLLM_CONCURRENCY одновременно (asyncio+httpx, семафор; батчит сам vLLM —
               continuous batching). Ответ приводится к контракту WAV 24k mono s16
               (перекодировка ffmpeg-ом при необходимости).
  inprocess  — фолбэк: qwen_tts в этом же процессе + динамический Batcher (путь v2,
               сохранён полностью).
Выбор: env TTS_ENGINE=auto|vllm|inprocess (default auto: пробуем vllm, при любой
ошибке его старта — WARNING в лог и фолбэк на inprocess).

env:
  AUTH_TOKEN      — обязателен (для режима сервера);
  VOICE_SAMPLE    — путь к сэмплу голоса (default /workspace/voice_sample.wav);
  MODEL_ID        — default Qwen/Qwen3-TTS-12Hz-1.7B-Base;
  PORT            — default 8000;
  REF_WINDOW_MODE — выбор окна референса из длинного сэмпла: lively (default) —
                    окно ~15 c с максимальной дисперсией громкости (живая речь
                    с интонационными перепадами); words — старый режим (окно
                    с максимумом слов);
  TTS_ENGINE      — auto|vllm|inprocess (default auto, см. «Движки» выше);
  VLLM_PORT (8091), VLLM_CONCURRENCY (32), VLLM_START_TIMEOUT_S (1200 = 20 мин),
  VLLM_READ_TIMEOUT_S (300) — параметры движка vllm;
  FFMPEG_BIN      — путь к ffmpeg (default: PATH; локальный selftest сам находит
                    bin/ffmpeg.exe проекта).

Фичи v4 (все ВЫКЛ по умолчанию — старый setup-путь работает как раньше;
в Dockerfile-образе их включает клиент через env):
  STRESS_MARKING=1 — разметка ударений перед синтезом (в ОБОИХ движках):
                    акцентор silero-stress («+» перед ударной гласной ->
                    акут U+0301 ПОСЛЕ гласной), словарь /workspace/stress_dict.yaml
                    либо /opt/app/stress_dict.yaml (слова, уже размеченные
                    клиентом/словарём, выигрывают у акцентора), фильтр
                    анти-«Оюрдо» (акут на «О»/«о» в НАЧАЛЕ слова не ставится).
                    Разметка НЕ попадает в тексты, с которыми сравнивает ASR-QC.
  ASR_QC=1        — ASR-самоконтроль: WAV чанка транскрибируется faster-whisper
                    (small, cuda -> cpu int8), CER против ЧИСТОГО текста чанка;
                    CER > ASR_QC_CER (0.15) -> пересинтез с джиттером температуры
                    (до ASR_QC_RETRIES (2) попыток), выбирается лучший CER.
                    Сбой ASR не ломает пайплайн (WARNING, WAV отдаётся как есть).
  DEADMAN=1       — сторож-«мертвец»: раз в 60 с проверка; аптайм процесса >
                    DEADMAN_MAX_HOURS (4.5) ч ЛИБО молчание клиента (нет POST /tts
                    и GET /result) > DEADMAN_IDLE_MIN (45) мин -> самоуничтожение
                    инстанса Vast: DELETE console.vast.ai по CONTAINER_API_KEY
                    (ограниченный ключ, Vast инжектит сам); id инстанса — из env
                    CONTAINER_ID / VAST_CONTAINERLABEL. Нет ключа/ид — WARNING,
                    сторож не активируется.

Тяжёлые зависимости (torch, qwen_tts, faster_whisper, silero_stress, num2words,
httpx) импортируются ТОЛЬКО внутри функций старта/движков/фич — модуль
импортируется и проходит selftest без GPU и без них.

Локальная проверка без GPU: python server.py --selftest
"""
import asyncio
import io
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

try:  # на Windows-консоли (локальный selftest) — UTF-8; на Linux безвредно
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------- настройки --

AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
PORT = int(os.environ.get("PORT", "8000"))
WORK_DIR = Path(os.environ.get("WORKSPACE", "/workspace"))
VOICE_SAMPLE = os.environ.get("VOICE_SAMPLE", str(WORK_DIR / "voice_sample.wav"))

SAMPLE_RATE = 24000        # частота итогового WAV (контракт API)
BATCH_SIZE = 4             # максимум запросов в одном батче генерации: чанки
                           # ~1800 симв (до 25 предложений) генерируются ~2-4 мин
                           # на 3090 — пачка из 8 держала бы VRAM и очередь
                           # вдвое дольше, пачка из 4 отдаёт результаты чаще
BATCH_WINDOW_S = 0.150     # ожидание после первого запроса, с
REF_MAX_FULL_S = 18.0      # сэмпл длиннее — вырезаем окно
REF_WINDOW_S = 15.0        # длина вырезаемого окна, с
# Режим выбора окна референса: lively (живая речь, макс. дисперсия громкости)
# либо words (старый режим — максимум слов в окне)
REF_WINDOW_MODE = os.environ.get("REF_WINDOW_MODE", "lively")
REF_RMS_FRAME_S = 0.050    # фрейм RMS для lively-режима, с
JOBS_CAP = 800             # максимум заданий в памяти (чанк ~1800 симв -> WAV
                           # ~2 мин ≈ 5-6 МБ; худший случай ~4.5 ГБ RAM — в лимит
                           # инстансов cpu_ram_min_gb=32 влезает с запасом)

# --- движки v3 ---
TTS_ENGINE = os.environ.get("TTS_ENGINE", "auto")  # auto|vllm|inprocess
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8091"))
VLLM_URL = "http://127.0.0.1:%d" % VLLM_PORT
VLLM_CONCURRENCY = int(os.environ.get("VLLM_CONCURRENCY", "32"))  # одновременных HTTP к vLLM
VLLM_START_TIMEOUT_S = int(os.environ.get("VLLM_START_TIMEOUT_S", "1200"))  # 20 мин на старт
VLLM_READ_TIMEOUT_S = int(os.environ.get("VLLM_READ_TIMEOUT_S", "300"))     # таймаут одной генерации
VLLM_LOG = WORK_DIR / "vllm.log"   # stdout+stderr сабпроцесса vLLM
VLLM_VOICE_NAME = "user_voice"     # имя голоса при регистрации способом (а)

# --- фичи v4 (env-флаги, по умолчанию ВЫКЛ — подробности в шапке модуля) ---
STRESS_MARKING = os.environ.get("STRESS_MARKING", "0") == "1"
STRESS_DICT_PATHS = (WORK_DIR / "stress_dict.yaml",
                     Path("/opt/app/stress_dict.yaml"))
ASR_QC = os.environ.get("ASR_QC", "0") == "1"
ASR_QC_CER = float(os.environ.get("ASR_QC_CER", "0.15"))
ASR_QC_RETRIES = int(os.environ.get("ASR_QC_RETRIES", "2"))
ASR_QC_BASE_TEMP = 0.9     # база джиттера температуры, если TTS_TEMPERATURE пуст
DEADMAN = os.environ.get("DEADMAN", "0") == "1"
DEADMAN_MAX_HOURS = float(os.environ.get("DEADMAN_MAX_HOURS", "4.5"))
DEADMAN_IDLE_MIN = float(os.environ.get("DEADMAN_IDLE_MIN", "45"))
DEADMAN_TOUCH_FILE = WORK_DIR / "last_poll"  # touch при каждом обращении клиента
DEADMAN_PERIOD_S = 60      # период проверки сторожа, с


def log(msg: str) -> None:
    """Лог в stdout с меткой времени (setup_server.sh перенаправляет в server.log)."""
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


# ------------------------------------------------------------ чистая логика --
# (эти функции не трогают GPU и проверяются в --selftest)

def pick_ref_window(words: list, window_s: float = REF_WINDOW_S) -> tuple:
    """Выбор окна ~window_s секунд с максимумом слов (скользящее окно по словам).

    words: [{"start": float, "end": float, "word": str}, ...] — по возрастанию start.
    Возвращает (t_start, t_end, text): границы окна в секундах и текст слов окна.
    """
    if not words:
        raise ValueError("пустой список слов — нечего выбирать")
    n = len(words)
    best_i, best_j, best_count = 0, 0, -1  # [i..j] включительно
    j = 0
    for i in range(n):
        if j < i:
            j = i
        # расширяем правую границу, пока слово целиком помещается в окно
        while j < n and words[j]["end"] - words[i]["start"] <= window_s:
            j += 1
        count = j - i
        if count > best_count:
            best_count, best_i, best_j = count, i, j - 1
    t0 = float(words[best_i]["start"])
    t1 = float(words[best_j]["end"])
    text = " ".join(w["word"].strip() for w in words[best_i:best_j + 1]).strip()
    return t0, t1, text


def pick_ref_window_lively(samples, sample_rate: int, window_s: float = REF_WINDOW_S,
                           frame_s: float = REF_RMS_FRAME_S) -> tuple:
    """«Живое» окно ~window_s секунд: максимальная дисперсия громкости.

    Громкость — RMS по фреймам frame_s (50 мс); окно из window_s/frame_s фреймов
    скользит с шагом в один фрейм, побеждает окно с максимальной дисперсией RMS:
    живая речь с интонационными перепадами (тихо/громко) вместо просто плотной
    или монотонной. Чистая функция: numpy-массив сэмплов -> границы.

    samples: массив сэмплов mono (float или int), sample_rate: частота, Гц.
    Возвращает (t_start, t_end) в секундах. Сэмпл короче окна -> весь целиком.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("пустой массив сэмплов — нечего выбирать")
    frame_len = max(1, int(round(frame_s * sample_rate)))
    n_frames = x.size // frame_len
    if n_frames < 2:
        return 0.0, x.size / float(sample_rate)  # короче двух фреймов — целиком
    rms = np.sqrt(np.mean(
        x[:n_frames * frame_len].reshape(n_frames, frame_len) ** 2, axis=1))
    win = int(round(window_s / frame_s))
    if win >= n_frames:
        # окно шире записи — берём всё, что нарезалось на целые фреймы
        return 0.0, n_frames * frame_len / float(sample_rate)
    # Скользящая дисперсия за O(n): var = E[r^2] - (E[r])^2 через кумулятивные суммы
    c1 = np.cumsum(np.concatenate(([0.0], rms)))
    c2 = np.cumsum(np.concatenate(([0.0], rms * rms)))
    mean = (c1[win:] - c1[:-win]) / win
    mean_sq = (c2[win:] - c2[:-win]) / win
    var = mean_sq - mean * mean
    best = int(np.argmax(var))
    t0 = best * frame_len / float(sample_rate)
    t1 = (best + win) * frame_len / float(sample_rate)
    return float(t0), float(t1)


def wav_bytes_from_array(samples, sample_rate: int = SAMPLE_RATE) -> bytes:
    """numpy-массив -> WAV bytes (mono, s16, RIFF) через стандартный модуль wave.

    float-массивы клиппятся в [-1, 1] и масштабируются в int16; int16 берётся как есть.
    """
    arr = np.asarray(samples)
    if arr.ndim > 1:
        arr = arr.reshape(-1)  # mono: плоский массив сэмплов
    if arr.dtype != np.int16:
        arr = np.clip(arr.astype(np.float64), -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(arr.tobytes())
    return buf.getvalue()


def wav_duration_s(path) -> float:
    """Длительность WAV-файла в секундах (стандартный wave, без ffprobe)."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def ffmpeg_bin() -> str:
    """Путь к ffmpeg: env FFMPEG_BIN -> PATH -> bin/ffmpeg.exe проекта (Windows selftest).

    На инстансе ffmpeg ставится setup_server.sh и виден в PATH; локальная ветка
    с bin/ffmpeg.exe нужна только для selftest перекодировки на Windows.
    """
    envp = os.environ.get("FFMPEG_BIN")
    if envp:
        return envp
    found = shutil.which("ffmpeg")
    if found:
        return found
    local = Path(__file__).resolve().parent.parent / "bin" / "ffmpeg.exe"
    if local.exists():
        return str(local)
    return "ffmpeg"


def wav_matches_contract(data: bytes) -> bool:
    """True, если байты — валидный WAV 24000 Гц mono s16 PCM (контракт API)."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return (w.getnchannels() == 1 and w.getsampwidth() == 2
                    and w.getframerate() == SAMPLE_RATE and w.getcomptype() == "NONE")
    except Exception:
        return False


def ensure_wav_contract(data: bytes) -> bytes:
    """Байты аудио-ответа vLLM -> контракт WAV 24k mono s16.

    Контрактный WAV возвращается как есть (тот же объект, без копий); всё прочее
    (другая частота/каналы, mp3/flac/ogg и т.п.) перекодируется ffmpeg-ом через
    временные файлы (пайпы не используем: wav-муксер в нессекаемый stdout пишет
    битые размеры RIFF-чанков).
    """
    if wav_matches_contract(data):
        return data
    src = tempfile.NamedTemporaryFile(prefix="tts_raw_", suffix=".bin", delete=False)
    dst_path = src.name + ".wav"
    try:
        src.write(data)
        src.close()
        _ffmpeg(["-i", src.name, "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-c:a", "pcm_s16le", "-f", "wav", dst_path])
        out = Path(dst_path).read_bytes()
    finally:
        for p in (src.name, dst_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    if not wav_matches_contract(out):
        raise RuntimeError("перекодировка ffmpeg не дала WAV 24k mono s16")
    return out


def build_vllm_payload(text: str, language: str, voice_mode: str,
                       ref_wav: str = "", ref_text: str = "",
                       temperature=None) -> dict:
    """JSON-тело запроса POST /v1/audio/speech к локальному vLLM (OpenAI-совместимое).

    voice_mode:
      "named"       — голос заранее зарегистрирован (способ (а)): voice=VLLM_VOICE_NAME;
      "per_request" — клон в каждом запросе (способ (б)): ref_audio=<локальный путь
                      к ref.wav> + ref_text (vLLM локальный — путь ему доступен).
    temperature (v4, ретраи ASR-QC): не None -> поле "temperature" в payload
    (сборка vLLM может его отвергнуть — vllm_synthesize повторит без него).
    """
    payload = {"model": MODEL_ID, "input": text, "language": language}
    if voice_mode == "named":
        payload["voice"] = VLLM_VOICE_NAME
    else:
        # vLLM-omni ждёт ref_audio как URI (file://...), а не голый путь — иначе
        # HTTP 400 "ref_audio must be a URL/base64/file URI" (поймано на живом тесте 22.07).
        from pathlib import Path as _P
        payload["ref_audio"] = (ref_wav if "://" in ref_wav
                                else _P(ref_wav).absolute().as_uri())
        payload["ref_text"] = ref_text
    if temperature is not None:
        payload["temperature"] = float(temperature)
    return payload


def choose_engine(requested: str, vllm_starter) -> str:
    """Логика выбора движка (чистая, проверяется в --selftest).

    requested: auto|vllm|inprocess (регистр не важен; неизвестное значение = auto
    с WARNING). vllm_starter — callable без аргументов: поднимает движок vllm,
    при провале бросает исключение.
    Возвращает "vllm" либо "inprocess". requested=vllm: ошибка старта
    пробрасывается наверх (явный выбор — фолбэка нет); auto: WARNING и inprocess.
    """
    req = (requested or "auto").strip().lower()
    if req not in ("auto", "vllm", "inprocess"):
        log("WARNING: TTS_ENGINE=%r неизвестен — трактую как auto" % requested)
        req = "auto"
    if req == "inprocess":
        return "inprocess"
    try:
        vllm_starter()
        return "vllm"
    except Exception as e:
        if req == "vllm":
            raise
        log("WARNING: старт vLLM не удался (%s: %s) — фолбэк на движок inprocess"
            % (type(e).__name__, e))
        return "inprocess"


# ------------------------------ разметка ударений (STRESS_MARKING, чистая часть) --

ACUTE = "\u0301"                     # комбинирующий акут: ставится ПОСЛЕ гласной
_RU_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"  # гласные, перед которыми акцентор ставит «+»
# «Слово» для словаря/разметки: буквы (рус/лат) + уже стоящие акуты
_WORD_RE = re.compile("[А-Яа-яЁёA-Za-z\u0301]+")


def strip_acute(text: str) -> str:
    """Убрать все акуты U+0301 (чистый текст для акцентора и для ASR-QC)."""
    return text.replace(ACUTE, "")


def convert_plus_to_acute(text: str) -> str:
    """Разметка акцентора «+гласная» -> «гласная»+U+0301 (акут ПОСЛЕ гласной).

    Фильтр анти-«Оюрдо»: если ударная гласная — «О»/«о» И это ПЕРВАЯ буква слова,
    акут НЕ ставится (известный баг Qwen: «О́рдо» читается как «Оюрдо»); «о» в
    середине/конце слова («окно́») размечается как обычно.
    «+» не перед русской гласной — не разметка, остаётся как есть.
    """
    out = []
    prev_is_letter = False  # была ли предыдущая выведенная позиция буквой слова
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "+" and i + 1 < n and text[i + 1] in _RU_VOWELS:
            vowel = text[i + 1]
            out.append(vowel)
            if not (vowel in "Оо" and not prev_is_letter):
                out.append(ACUTE)
            prev_is_letter = True
            i += 2
            continue
        out.append(ch)
        prev_is_letter = ch.isalpha()
        i += 1
    return "".join(out)


def parse_stress_dict(text: str) -> dict:
    """Разбор stress_dict.yaml (формат «слово: "замена"», см. config/stress_dict.yaml).

    Сначала PyYAML (если установлен), при отсутствии/сбое — построчный разбор:
    формат словаря нарочно плоский, вложенных структур нет.
    """
    try:
        import yaml
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition(":")
        if not sep:
            continue
        val = val.split("#", 1)[0].strip().strip("\"'")
        key = key.strip().strip("\"'")
        if key and val:
            result[key] = val
    return result


def apply_stress_dict(text: str, stress_dict: dict) -> str:
    """Замены словаря ударений: слово ЦЕЛИКОМ, регистронезависимо; замена — ровно
    как написана в словаре. Слова, уже несущие акут (клиент разметил), не трогаем."""
    if not stress_dict:
        return text
    lower_map = {strip_acute(k).lower(): v for k, v in stress_dict.items()}

    def _repl(m):
        w = m.group(0)
        if ACUTE in w:
            return w
        return lower_map.get(w.lower(), w)

    return _WORD_RE.sub(_repl, text)


def mark_stress(text: str, accentor=None, stress_dict=None) -> str:
    """Полная разметка ударений текста чанка перед синтезом (чистая функция).

    Шаги: (а) словарь + запоминаем слова, УЖЕ несущие акут (клиентская/словарная
    разметка выигрывает); (б) акцентор на тексте без акутов («+» перед ударной
    гласной); (в)+(г) конвертер «+»->акут с фильтром анти-«Оюрдо»; (д) возврат
    слов из (а) на их места в словарной разметке. Ошибка акцентора не фатальна:
    WARNING и текст без авторазметки (словарь уже применён).
    """
    text = apply_stress_dict(text, stress_dict or {})
    if accentor is None:
        return text
    tokens = _WORD_RE.findall(text)
    preserved = {i: w for i, w in enumerate(tokens) if ACUTE in w}
    clean = strip_acute(text)
    try:
        plussed = accentor(clean)
        if not isinstance(plussed, str):
            raise TypeError("акцентор вернул %s вместо str" % type(plussed).__name__)
    except Exception as e:
        log("WARNING: акцентор упал (%s: %s) — чанк остаётся без авторазметки"
            % (type(e).__name__, e))
        return text
    accented = convert_plus_to_acute(plussed)
    if not preserved:
        return accented
    out_tokens = _WORD_RE.findall(accented)
    if len(out_tokens) == len(tokens):
        counter = [0]

        def _restore(m):
            i = counter[0]
            counter[0] += 1
            return preserved.get(i, m.group(0))

        return _WORD_RE.sub(_restore, accented)
    # Акцентор изменил число слов (не должен) — деградация: возврат по формам
    log("WARNING: разметка: число слов изменилось (%d -> %d) — словарные слова "
        "возвращаю по совпадению форм" % (len(tokens), len(out_tokens)))
    by_bare = {strip_acute(w).lower(): w for w in preserved.values()}

    def _restore_by_form(m):
        w = m.group(0)
        return by_bare.get(strip_acute(w).lower(), w)

    return _WORD_RE.sub(_restore_by_form, accented)


# ------------------------------------ ASR-самоконтроль (ASR_QC, чистая часть) --

_NUM2WORDS = None  # ленивый кэш num2words: None = не пробовали, False = пакета нет


def _digits_to_words_ru(digits: str) -> str:
    """«5» -> «пять» через num2words (лениво); без пакета цифры остаются как есть."""
    global _NUM2WORDS
    if _NUM2WORDS is None:
        try:
            from num2words import num2words
            _NUM2WORDS = num2words
        except ImportError:
            _NUM2WORDS = False
            log("WARNING: num2words не установлен — цифры в транскрипте ASR "
                "останутся цифрами (возможен ложный штраф CER)")
    if not _NUM2WORDS:
        return digits
    try:
        return _NUM2WORDS(int(digits), lang="ru")
    except Exception:
        return digits


def normalize_for_asr(text: str) -> str:
    """Нормализация ОБЕИХ сторон перед CER: без акутов, lower, ё->е, цифры->слова
    (в исходнике цифр уже нет — клиент нормализует; это для транскрипта Whisper),
    без пунктуации, схлопнутые пробелы."""
    t = strip_acute(text).lower().replace("ё", "е")
    t = re.sub(r"\d+", lambda m: " %s " % _digits_to_words_ru(m.group(0)), t)
    t = re.sub(r"[^0-9a-zа-я]+", " ", t)
    return " ".join(t.split())


def levenshtein(a: str, b: str) -> int:
    """Расстояние Левенштейна (замена/вставка/удаление) без новых зависимостей.

    O(len·len) время, O(len) память; внутренняя строка DP считается numpy-полосой
    (цепочка вставок слева направо — через min-accumulate: cur[j] =
    min_k<=j(cand[k] + (j - k))), чанк ~1500 симв — миллисекунды."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    b_codes = np.array([ord(c) for c in b], dtype=np.int64)
    js = np.arange(len(b) + 1, dtype=np.int64)
    prev = js.copy()
    cand = np.empty(len(b) + 1, dtype=np.int64)
    for i, ca in enumerate(a, 1):
        cand[0] = i
        cand[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b_codes != ord(ca)))
        prev = np.minimum.accumulate(cand - js) + js
    return int(prev[-1])


def cer(ref: str, hyp: str) -> float:
    """CER = расстояние Левенштейна / длина эталона (пустой эталон -> длина гипотезы)."""
    return levenshtein(ref, hyp) / float(max(1, len(ref)))


def qc_retry_temps(base_temp: float, retries: int) -> list:
    """Лестница температур ретраев ASR-QC: temp*1.05, temp*0.95, дальше шире."""
    mults = (1.05, 0.95, 1.10, 0.90, 1.15, 0.85)
    return [round(base_temp * mults[k % len(mults)], 4)
            for k in range(max(0, int(retries)))]


# ------------------------------------ сторож-«мертвец» (DEADMAN, чистая часть) --

def parse_instance_id(env) -> "int | None":
    """id инстанса Vast из env контейнера: CONTAINER_ID либо VAST_CONTAINERLABEL
    (формат «C.12345») — берётся первое найденное число; нет нигде — None."""
    for var in ("CONTAINER_ID", "VAST_CONTAINERLABEL"):
        m = re.search(r"\d+", str(env.get(var) or ""))
        if m:
            return int(m.group(0))
    return None


def deadman_decision(uptime_s: float, idle_s: float, max_hours: float,
                     idle_min: float) -> "str | None":
    """Решение сторожа (чистая функция, dry-run в selftest): причина
    самоуничтожения (str) либо None (живём). Ровно на лимите — ещё живём."""
    if uptime_s > max_hours * 3600.0:
        return "аптайм %.2f ч превысил лимит %.2f ч" % (uptime_s / 3600.0, max_hours)
    if idle_s > idle_min * 60.0:
        return "клиент молчит %.1f мин (лимит %.1f мин)" % (idle_s / 60.0, idle_min)
    return None


class Batcher:
    """Динамический батчер: копит запросы в asyncio.Queue и генерирует пачками.

    Фоновая корутина run() забирает первый запрос, ждёт до BATCH_WINDOW_S секунд,
    добирая до batch_size запросов, и отдаёт пачку generate_fn в ThreadPoolExecutor(1)
    (torch блокирующий, GPU один — один воркер генерации). Результаты раздаются
    ожидающим Future; ошибка генерации уходит во все Future пачки.
    """

    def __init__(self, generate_fn, batch_size: int = BATCH_SIZE,
                 window_s: float = BATCH_WINDOW_S):
        self.generate_fn = generate_fn      # (texts: list[str], langs: list[str]) -> (wavs, sr)
        self.batch_size = batch_size
        self.window_s = window_s
        self.queue: asyncio.Queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)  # единственный GPU-воркер
        self.inflight = 0                   # запросов сейчас в генерации (для /health)

    def pending(self) -> int:
        """Запросов в очереди + в текущей генерации."""
        return self.queue.qsize() + self.inflight

    async def submit(self, text: str, language: str):
        """Поставить запрос в очередь и дождаться результата: (samples, sample_rate)."""
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((text, language, fut))
        return await fut

    async def stop(self) -> None:
        """Попросить run() завершиться (сентинел в очередь)."""
        await self.queue.put(None)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await self.queue.get()
            if item is None:
                return
            batch = [item]
            stopping = False
            deadline = loop.time() + self.window_s
            # добираем запросы до полного батча либо до истечения окна
            while len(batch) < self.batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if nxt is None:
                    stopping = True
                    break
                batch.append(nxt)
            texts = [b[0] for b in batch]
            langs = [b[1] for b in batch]
            self.inflight = len(batch)
            try:
                wavs, sr = await loop.run_in_executor(
                    self.executor, self.generate_fn, texts, langs)
                if len(wavs) != len(texts):
                    # Иначе zip молча отбросил бы хвост, и его Future зависли бы навсегда
                    raise RuntimeError(
                        "generate_voice_clone вернул %d wav на %d текстов — "
                        "количество результатов не совпадает с размером батча"
                        % (len(wavs), len(texts)))
                for (_, _, fut), samples in zip(batch, wavs):
                    if not fut.done():
                        fut.set_result((samples, sr))
            except Exception as e:  # ошибка генерации -> всем ожидающим пачки
                for _, _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
            finally:
                self.inflight = 0
            if stopping:
                return


# ----------------------------------------------------------- состояние/старт --

class ServerState:
    """Глобальное состояние сервера (движок, модель, готовность, ошибка старта)."""

    def __init__(self):
        self.ready = False
        self.error = None           # str | None — попадает в /health
        self.engine = None          # "vllm" | "inprocess" (выбирается на старте)
        self.model = None           # Qwen3TTSModel (движок inprocess)
        self.prompt_items = None    # результат create_voice_clone_prompt (inprocess)
        self.batcher = None         # Batcher (inprocess)
        self.ref_wav = None         # путь к подготовленному ref.wav
        self.ref_text = None        # транскрипт референса
        self.vllm_proc = None       # subprocess.Popen сабпроцесса vLLM
        self.vllm_voice_mode = None  # "named" | "per_request" (движок vllm)
        self.vllm_client = None     # httpx.AsyncClient (создаётся в event loop)
        self.vllm_sem = None        # asyncio.Semaphore(VLLM_CONCURRENCY)
        # --- фичи v4 (env-флаги, см. шапку модуля) ---
        self.accentor = None        # silero-stress акцентор (STRESS_MARKING=1)
        self.stress_dict = {}       # словарь ударений {слово: слово с акутом}
        self.asr_model = None       # faster-whisper small для ASR-QC (лениво)
        self.asr_lock = threading.Lock()   # загрузка + сериализация транскрипций
        self.asr_disabled = False   # True после провала импорта/загрузки ASR
        self.started_at = time.time()      # старт процесса (мертвец: аптайм)
        self.last_client_ts = time.time()  # последнее обращение клиента (idle)


STATE = ServerState()


def _ffmpeg(args: list) -> None:
    """Запуск ffmpeg (путь через ffmpeg_bin(); на инстансе — из PATH) с проверкой кода."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y"] + args
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: %s | stderr: %s"
                           % (" ".join(args), (r.stderr or "").strip()[-500:]))


def _read_wav_samples(path) -> tuple:
    """WAV (mono s16) -> (numpy float64 в [-1, 1], sample_rate).

    Читает ref_full.wav, уже приведённый ffmpeg-ом к mono 24k s16 —
    массив уходит в pick_ref_window_lively.
    """
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return arr, sr


def _window_words(words: list, t0: float, t1: float):
    """Слова, чья середина попала в [t0, t1] -> (t0', t1', text) либо None.

    Границы окна подтягиваются к границам крайних слов, чтобы ffmpeg не резал
    слово посередине и REF_TEXT точно соответствовал вырезанному аудио.
    words — по возрастанию start (как отдаёт whisper).
    """
    sel = [w for w in words if t0 <= (w["start"] + w["end"]) / 2.0 <= t1]
    if not sel:
        return None
    text = " ".join(w["word"].strip() for w in sel).strip()
    return float(sel[0]["start"]), float(sel[-1]["end"]), text


def prepare_reference() -> tuple:
    """Шаги 1-3 старта: перекодировка сэмпла, транскрипция, выбор окна ~15 с.

    Окно выбирается по env REF_WINDOW_MODE: lively (default) — участок с
    максимальной дисперсией громкости (живая интонация); words — старый режим
    (максимум слов). Возвращает (путь_к_ref_wav, ref_text) для
    create_voice_clone_prompt.
    """
    ref_full = WORK_DIR / "ref_full.wav"

    # 1. Сэмпл голоса -> mono 24 кГц s16 (формат, который ждёт модель)
    log("Старт: перекодирую сэмпл %s -> %s" % (VOICE_SAMPLE, ref_full))
    if not Path(VOICE_SAMPLE).exists():
        raise FileNotFoundError("сэмпл голоса не найден: %s" % VOICE_SAMPLE)
    _ffmpeg(["-i", VOICE_SAMPLE, "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-c:a", "pcm_s16le", str(ref_full)])
    duration = wav_duration_s(ref_full)
    log("Сэмпл: %.1f c" % duration)

    # 2. Транскрипция faster-whisper с пословными таймстемпами.
    # Устройство из env REF_WHISPER_DEVICE (default cpu): база vllm-omni на CUDA 13,
    # а ctranslate2-колёса ждут libcublas.so.12 (CUDA 12) — на GPU падает. CPU int8
    # надёжен и для короткого сэмпла (~15-150 c) быстр (одноразово на старте сервера).
    dev = os.environ.get("REF_WHISPER_DEVICE", "cpu")
    ct = "float16" if dev == "cuda" else "int8"
    log("Транскрибирую сэмпл (faster-whisper small, %s, %s)..." % (dev, ct))
    from faster_whisper import WhisperModel  # тяжёлый импорт — только здесь
    wm = WhisperModel("small", device=dev, compute_type=ct)
    segments, _info = wm.transcribe(str(ref_full), language="ru",
                                    word_timestamps=True)
    words = []
    for seg in segments:  # генератор — итерация и запускает распознавание
        for w in (seg.words or []):
            token = (w.word or "").strip()
            if token:
                words.append({"start": float(w.start), "end": float(w.end),
                              "word": token})
    # Освобождаем VRAM под TTS-модель
    del wm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    if not words:
        raise RuntimeError("whisper не распознал ни одного слова в сэмпле голоса")
    full_text = " ".join(w["word"] for w in words)
    log("Распознано слов: %d" % len(words))

    # 3. Длинный сэмпл -> окно ~15 c (режим из env REF_WINDOW_MODE); короткий — целиком
    if duration > REF_MAX_FULL_S:
        picked = None
        if REF_WINDOW_MODE != "words":  # lively (и любое другое значение) = живое окно
            samples, sr = _read_wav_samples(ref_full)
            lt0, lt1 = pick_ref_window_lively(samples, sr, REF_WINDOW_S)
            picked = _window_words(words, lt0, lt1)
            if picked is None:  # в живом окне ни одного слова — деградируем в words
                log("ВНИМАНИЕ: в lively-окне %.2f-%.2f c нет слов, "
                    "переключаюсь на режим words" % (lt0, lt1))
        if picked is None:
            picked = pick_ref_window(words, REF_WINDOW_S)
        t0, t1, ref_text = picked
        ref = WORK_DIR / "ref.wav"
        log("Вырезаю окно %.2f-%.2f c (режим %s, %d слов) -> %s"
            % (t0, t1, REF_WINDOW_MODE, len(ref_text.split()), ref))
        _ffmpeg(["-i", str(ref_full), "-ss", "%.3f" % t0, "-to", "%.3f" % t1,
                 "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
                 str(ref)])
        return str(ref), ref_text
    return str(ref_full), full_text


def load_model_and_warmup(ref_wav: str, ref_text: str) -> None:
    """Шаг 4 старта: загрузка Qwen3-TTS, клон-промпт, прогревочная генерация."""
    log("Загружаю модель %s ..." % MODEL_ID)
    import torch                      # тяжёлые импорты — только здесь
    from qwen_tts import Qwen3TTSModel
    # БЕЗ flash_attention_2 (не собираем flash-attn, дефолтное внимание)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16)
    # Стиль голоса (по итогам прослушки 19.07): TTS_XVEC_ONLY=1 — «тембр-только»
    # (манера модели, живее ровного референса); 0 — полный клон манеры.
    xvec_only = os.environ.get("TTS_XVEC_ONLY", "0") == "1"
    log("Клон-промпт: x_vector_only_mode=%s" % xvec_only)
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_wav, ref_text=ref_text, x_vector_only_mode=xvec_only)
    STATE.model = model
    STATE.prompt_items = prompt_items
    log("Прогревочная генерация...")
    model.generate_voice_clone(text=["Проверка связи."], language=["Russian"],
                               voice_clone_prompt=prompt_items)
    log("Модель готова.")


def generate_batch(texts: list, langs: list):
    """Блокирующая генерация пачки (вызывается батчером в ThreadPoolExecutor(1))."""
    vc = STATE.prompt_items
    # create_voice_clone_prompt может вернуть список из одного элемента —
    # для батча дублируем его на каждый текст
    if isinstance(vc, list) and len(vc) == 1 and len(texts) > 1:
        vc = vc * len(texts)
    kwargs = dict(text=texts, language=langs, voice_clone_prompt=vc)
    # Температура стиля (TTS_TEMPERATURE, пусто = дефолт модели). Если сборка
    # qwen_tts не принимает параметры генерации — тихо откатываемся к дефолту.
    temp = os.environ.get("TTS_TEMPERATURE", "").strip()
    if temp:
        try:
            wavs, sr = STATE.model.generate_voice_clone(
                **kwargs, temperature=float(temp), top_p=0.95)
            return wavs, sr
        except TypeError:
            log("WARNING: generate_voice_clone не принял temperature — дефолт")
    wavs, sr = STATE.model.generate_voice_clone(**kwargs)
    return wavs, sr


def generate_single_with_temperature(text: str, lang: str, temperature: float):
    """Одиночный пересинтез с явной температурой (ретраи ASR-QC, движок inprocess).

    Запускается в ТОМ ЖЕ ThreadPoolExecutor(1), что и батчер — GPU не делится.
    Сборка qwen_tts без параметров генерации -> WARNING и дефолт (как в generate_batch).
    """
    kwargs = dict(text=[text], language=[lang],
                  voice_clone_prompt=STATE.prompt_items)
    try:
        return STATE.model.generate_voice_clone(
            **kwargs, temperature=float(temperature), top_p=0.95)
    except TypeError:
        log("WARNING: generate_voice_clone не принял temperature — ретрай дефолтом")
        return STATE.model.generate_voice_clone(**kwargs)


# ------------------------------------------------------------- движок vllm --

def _vllm_log_tail(n: int = 30) -> str:
    """Последние строки vllm.log — в текст ошибки при падении старта."""
    try:
        lines = VLLM_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(vllm.log недоступен)"


def _kill_vllm() -> None:
    """Гасим сабпроцесс vLLM (вся группа процессов: у vLLM есть воркеры-дети).

    Обязательно перед фолбэком на inprocess — иначе VRAM останется занята.
    """
    proc = STATE.vllm_proc
    if proc is None:
        return
    if proc.poll() is None:
        log("vLLM: останавливаю сабпроцесс pid=%d" % proc.pid)
        try:
            if hasattr(os, "killpg"):  # Linux: убиваем группу целиком
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    STATE.vllm_proc = None


def _wait_vllm_ready(proc) -> None:
    """Поллинг готовности vLLM (GET /health) до VLLM_START_TIMEOUT_S.

    Модель качается с HuggingFace и компилируется — старт занимает минуты
    (лимит 20 мин). Умерший сабпроцесс детектим сразу, с хвостом vllm.log.
    """
    deadline = time.time() + VLLM_START_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                "vLLM-процесс завершился на старте (код %s). Хвост vllm.log:\n%s"
                % (proc.returncode, _vllm_log_tail()))
        try:
            with urllib.request.urlopen(VLLM_URL + "/health", timeout=5) as resp:
                if resp.status == 200:
                    log("vLLM: /health отвечает 200 — сервер поднялся")
                    return
        except Exception:
            pass  # ещё не слушает порт — ждём дальше
        time.sleep(5)
    raise RuntimeError("vLLM не поднялся за %d с (хвост vllm.log:\n%s)"
                       % (VLLM_START_TIMEOUT_S, _vllm_log_tail()))


def _register_voice_named(client, ref_wav: str, ref_text: str) -> None:
    """Способ (а): именованная регистрация голоса POST /v1/audio/voices (multipart)."""
    with open(ref_wav, "rb") as f:
        audio = f.read()
    r = client.post("/v1/audio/voices",
                    files={"audio_sample": (Path(ref_wav).name, audio, "audio/wav")},
                    data={"name": VLLM_VOICE_NAME, "ref_text": ref_text})
    if r.status_code not in (200, 201):
        raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:300]))
    log("vLLM: голос '%s' зарегистрирован (способ (а))" % VLLM_VOICE_NAME)


def _vllm_speech_sync(client, payload: dict) -> bytes:
    """Синхронный POST /v1/audio/speech (для прогрева на старте) -> аудио-байты."""
    r = client.post("/v1/audio/speech", json=payload)
    if r.status_code != 200:
        raise RuntimeError("vLLM /v1/audio/speech: HTTP %d: %s"
                           % (r.status_code, r.text[:300]))
    if not r.content:
        raise RuntimeError("vLLM вернул пустое тело ответа")
    return bytes(r.content)


def start_vllm_engine(ref_wav: str, ref_text: str) -> None:
    """Полный старт движка vllm: сабпроцесс -> готовность -> голос -> прогрев.

    Любая ошибка -> исключение (choose_engine решает: фолбэк на inprocess или фатал);
    сабпроцесс при этом гасится, VRAM освобождается. Успех: STATE.vllm_proc и
    STATE.vllm_voice_mode заполнены.
    """
    import httpx  # лениво: ставится setup_server.sh, локально для selftest не нужен
    # vllm живёт в отдельном venv (transformers>=5), путь к бинарю — env VLLM_BIN;
    # фолбэк "vllm" — если venv не создавался (старые инстансы).
    vllm_bin = os.environ.get("VLLM_BIN") or "vllm"
    if "/" in vllm_bin and not Path(vllm_bin).exists():
        log("vLLM: бинарь %s не найден — пробую 'vllm' из PATH" % vllm_bin)
        vllm_bin = "vllm"
    log("vLLM: запускаю сабпроцесс: %s serve %s --omni --port %d (лог: %s)"
        % (vllm_bin, MODEL_ID, VLLM_PORT, VLLM_LOG))
    logf = open(VLLM_LOG, "ab")
    try:
        proc = subprocess.Popen(
            [vllm_bin, "serve", MODEL_ID, "--omni", "--port", str(VLLM_PORT)],
            stdout=logf, stderr=subprocess.STDOUT, cwd=str(WORK_DIR),
            start_new_session=True)  # своя группа — гасим вместе с воркерами-детьми
    finally:
        logf.close()  # дескриптор унаследован сабпроцессом
    STATE.vllm_proc = proc
    try:
        _wait_vllm_ready(proc)
        timeout = httpx.Timeout(connect=15.0, read=float(VLLM_READ_TIMEOUT_S),
                                write=60.0, pool=None)
        with httpx.Client(base_url=VLLM_URL, timeout=timeout) as client:
            # Способ (а) — именованный голос; любой сбой -> способ (б) ref_audio в запросе
            mode = "named"
            try:
                _register_voice_named(client, ref_wav, ref_text)
            except Exception as e:
                log("WARNING: vLLM: регистрация голоса не удалась (%s: %s) — "
                    "фолбэк: ref_audio в каждом запросе (способ (б))"
                    % (type(e).__name__, e))
                mode = "per_request"
            # Прогрев выбранного способа + проверка формата ответа
            payload = build_vllm_payload("Проверка связи.", "Russian", mode,
                                         ref_wav, ref_text)
            try:
                raw = _vllm_speech_sync(client, payload)
            except Exception as e:
                if mode != "named":
                    raise
                # голос зарегистрировался, но не генерирует — пробуем способ (б)
                log("WARNING: vLLM: прогрев с именованным голосом не удался "
                    "(%s: %s) — фолбэк: ref_audio в запросе" % (type(e).__name__, e))
                mode = "per_request"
                payload = build_vllm_payload("Проверка связи.", "Russian", mode,
                                             ref_wav, ref_text)
                raw = _vllm_speech_sync(client, payload)
            data = ensure_wav_contract(raw)
            if data is raw:
                log("vLLM: прогрев OK, ответ сразу в контракте WAV 24k mono s16 "
                    "(%d байт)" % len(data))
            else:
                log("vLLM: прогрев OK, ответ перекодируется ffmpeg-ом в контракт "
                    "(%d -> %d байт)" % (len(raw), len(data)))
        STATE.vllm_voice_mode = mode
        log("vLLM: движок готов (voice_mode=%s, concurrency=%d)"
            % (mode, VLLM_CONCURRENCY))
    except Exception:
        _kill_vllm()  # не оставляем полуживой vLLM держать VRAM
        raise


async def vllm_synthesize(text: str, language: str, temperature=None) -> bytes:
    """Одно задание через vLLM: семафор -> POST /v1/audio/speech -> WAV контракта.

    До VLLM_CONCURRENCY одновременных запросов; батчирует сам vLLM (continuous
    batching). Перекодировка (если нужна) — блокирующий ffmpeg, уводим в executor.
    temperature (ретраи ASR-QC): поле в payload; если сборка vLLM его отвергла —
    WARNING и повтор без него.
    """
    import httpx
    if STATE.vllm_client is None:  # event loop однопоточный — гонки нет
        STATE.vllm_client = httpx.AsyncClient(
            base_url=VLLM_URL,
            timeout=httpx.Timeout(connect=15.0, read=float(VLLM_READ_TIMEOUT_S),
                                  write=60.0, pool=None),
            limits=httpx.Limits(max_connections=VLLM_CONCURRENCY + 4))
    async with STATE.vllm_sem:
        payload = build_vllm_payload(text, language, STATE.vllm_voice_mode,
                                     STATE.ref_wav, STATE.ref_text, temperature)
        r = await STATE.vllm_client.post("/v1/audio/speech", json=payload)
        if r.status_code != 200 and temperature is not None:
            log("WARNING: vLLM отверг temperature=%.3f (HTTP %d) — повтор без неё"
                % (temperature, r.status_code))
            payload = build_vllm_payload(text, language, STATE.vllm_voice_mode,
                                         STATE.ref_wav, STATE.ref_text)
            r = await STATE.vllm_client.post("/v1/audio/speech", json=payload)
    if r.status_code != 200:
        raise RuntimeError("vLLM /v1/audio/speech: HTTP %d: %s"
                           % (r.status_code, r.text[:300]))
    data = bytes(r.content)
    if not data:
        raise RuntimeError("vLLM вернул пустое тело ответа")
    return await asyncio.get_running_loop().run_in_executor(
        None, ensure_wav_contract, data)


def startup_worker() -> None:
    """Фоновый поток старта: ready=False до самого конца, любая ошибка -> /health.

    Порядок: референс (ffmpeg + whisper, общий для обоих движков) -> выбор движка
    (TTS_ENGINE: auto пробует vllm и при ошибке откатывается на inprocess).
    """
    try:
        init_features()  # лёгкая инициализация фич v4 (ошибки не фатальны)
        ref_wav, ref_text = prepare_reference()
        STATE.ref_wav, STATE.ref_text = ref_wav, ref_text
        engine = choose_engine(TTS_ENGINE,
                               lambda: start_vllm_engine(ref_wav, ref_text))
        if engine == "inprocess":
            load_model_and_warmup(ref_wav, ref_text)
        STATE.engine = engine
        STATE.ready = True
        log("Сервер готов (ready=true, engine=%s)." % engine)
    except Exception as e:
        STATE.error = "%s: %s" % (type(e).__name__, e)
        log("ОШИБКА старта: %s" % STATE.error)
        traceback.print_exc()


# ------------------------------------------- фичи v4: runtime (см. шапку модуля) --

def _load_stress_dict() -> dict:
    """Словарь ударений с диска: первый существующий из STRESS_DICT_PATHS."""
    for p in STRESS_DICT_PATHS:
        try:
            p = Path(p)
            if p.exists():
                d = parse_stress_dict(p.read_text(encoding="utf-8"))
                log("Разметка ударений: словарь %s — %d слов" % (p, len(d)))
                return d
        except Exception as e:
            log("WARNING: словарь ударений %s не прочитан (%s: %s)"
                % (p, type(e).__name__, e))
    return {}


def init_features() -> None:
    """Лёгкая инициализация фич v4 на старте (любая ошибка — WARNING, не фатал).

    Разметка ударений: словарь + акцентор silero-stress (ленивый импорт; вызовы
    акцентора сериализуются локом — torch-модель не обязана быть потокобезопасной).
    ASR-QC: только проверка импорта faster_whisper (модель грузится лениво).
    """
    if STRESS_MARKING:
        STATE.stress_dict = _load_stress_dict()
        try:
            from silero_stress import load_accentor  # ленивый импорт (внутри torch)
            raw_accentor = load_accentor()
            acc_lock = threading.Lock()

            def locked_accentor(t, _a=raw_accentor, _l=acc_lock):
                with _l:
                    return _a(t)

            STATE.accentor = locked_accentor
            log("Разметка ударений: акцентор silero-stress загружен")
        except Exception as e:
            STATE.accentor = None
            log("WARNING: silero_stress недоступен (%s: %s) — авторазметка "
                "выключена, работает только словарь" % (type(e).__name__, e))
    if ASR_QC:
        try:
            import faster_whisper  # noqa: F401 — только проверка наличия пакета
            log("ASR-QC включён: порог CER=%.2f, ретраев=%d (whisper small, лениво)"
                % (ASR_QC_CER, ASR_QC_RETRIES))
        except ImportError:
            STATE.asr_disabled = True
            log("WARNING: ASR_QC=1, но faster_whisper не установлен — фича отключена")


async def prepare_tts_text(clean_text: str) -> str:
    """Текст для синтеза: разметка ударений (STRESS_MARKING) поверх чистого.

    Чистый текст остаётся в job["text"] — именно с ним сравнивает ASR-QC:
    разметка в тексты для сравнения не попадает."""
    if not STRESS_MARKING or (STATE.accentor is None and not STATE.stress_dict):
        return clean_text
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, mark_stress, clean_text,
                                          STATE.accentor, STATE.stress_dict)
    except Exception as e:
        log("WARNING: разметка ударений упала (%s: %s) — чанк без разметки"
            % (type(e).__name__, e))
        return clean_text


def _ensure_asr_model() -> bool:
    """Однократная ленивая загрузка faster-whisper small: cuda -> фолбэк cpu int8."""
    if STATE.asr_disabled:
        return False
    if STATE.asr_model is not None:
        return True
    with STATE.asr_lock:
        if STATE.asr_model is not None:
            return True
        if STATE.asr_disabled:
            return False
        try:
            from faster_whisper import WhisperModel
            try:
                m = WhisperModel("small", device="cuda", compute_type="float16")
                log("ASR-QC: faster-whisper small загружен (cuda, float16)")
            except Exception as e:
                log("ASR-QC: cuda недоступна (%s) — пробую cpu int8" % e)
                m = WhisperModel("small", device="cpu", compute_type="int8")
                log("ASR-QC: faster-whisper small загружен (cpu, int8)")
            STATE.asr_model = m
            return True
        except Exception as e:
            STATE.asr_disabled = True
            log("WARNING: ASR-QC отключён — модель не загрузилась (%s: %s)"
                % (type(e).__name__, e))
            return False


def _qc_cer_sync(clean_text: str, wav_bytes: bytes):
    """Транскрипция WAV (ru) + CER против чистого текста; None при любом сбое ASR."""
    if not _ensure_asr_model():
        return None
    try:
        with STATE.asr_lock:  # модель одна — транскрипции сериализуем
            segments, _info = STATE.asr_model.transcribe(io.BytesIO(wav_bytes),
                                                         language="ru")
            transcript = " ".join((seg.text or "").strip() for seg in segments)
        return cer(normalize_for_asr(clean_text), normalize_for_asr(transcript))
    except Exception as e:
        log("WARNING: ASR-QC транскрипция упала (%s: %s) — чанк без контроля"
            % (type(e).__name__, e))
        return None


def _qc_base_temperature() -> float:
    """База температуры для ретраев: env TTS_TEMPERATURE либо ASR_QC_BASE_TEMP."""
    t = os.environ.get("TTS_TEMPERATURE", "").strip()
    try:
        return float(t) if t else ASR_QC_BASE_TEMP
    except ValueError:
        return ASR_QC_BASE_TEMP


async def synthesize_once(text: str, language: str, temperature=None) -> bytes:
    """Один синтез выбранным движком -> WAV-байты контракта.

    temperature — переопределение для ретраев ASR-QC: vllm — поле в payload
    (см. vllm_synthesize); inprocess — одиночный вызов в ТОМ ЖЕ
    ThreadPoolExecutor(1) батчера (GPU-доступ сериализован)."""
    if STATE.engine == "vllm":
        return await vllm_synthesize(text, language, temperature)
    if temperature is None:
        samples, sr = await STATE.batcher.submit(text, language)
        return wav_bytes_from_array(samples, sr)
    loop = asyncio.get_running_loop()
    wavs, sr = await loop.run_in_executor(
        STATE.batcher.executor, generate_single_with_temperature,
        text, language, temperature)
    return wav_bytes_from_array(wavs[0], sr)


async def synth_with_qc(job_id: str, clean_text: str, tts_text: str,
                        language: str) -> bytes:
    """Синтез чанка + ASR-самоконтроль (если ASR_QC=1).

    CER > ASR_QC_CER -> пересинтез с джиттером температуры (qc_retry_temps),
    выбирается попытка с ЛУЧШИМ CER. Сравнение — с чистым текстом (без разметки).
    Любой сбой ASR не ломает пайплайн: WARNING и WAV отдаётся как есть.
    """
    data = await synthesize_once(tts_text, language)
    if not ASR_QC or STATE.asr_disabled:
        return data
    loop = asyncio.get_running_loop()
    c = await loop.run_in_executor(None, _qc_cer_sync, clean_text, data)
    if c is None:
        return data
    if c <= ASR_QC_CER:
        log("ASR-QC %s: CER %.3f <= %.2f — OK" % (job_id, c, ASR_QC_CER))
        return data
    cers = ["%.3f" % c]
    best_cer, best_data = c, data
    for temp in qc_retry_temps(_qc_base_temperature(), ASR_QC_RETRIES):
        try:
            data2 = await synthesize_once(tts_text, language, temp)
        except Exception as e:
            log("WARNING: ASR-QC %s: пересинтез (temp=%.3f) упал (%s: %s) — "
                "беру лучшее из имеющегося" % (job_id, temp, type(e).__name__, e))
            break
        c2 = await loop.run_in_executor(None, _qc_cer_sync, clean_text, data2)
        if c2 is None:
            break
        cers.append("%.3f" % c2)
        if c2 < best_cer:
            best_cer, best_data = c2, data2
        if c2 <= ASR_QC_CER:
            break
    log("ASR-QC %s: CER попыток [%s] -> выбрана лучшая (CER %.3f, порог %.2f)"
        % (job_id, ", ".join(cers), best_cer, ASR_QC_CER))
    return best_data


def _note_client_activity() -> None:
    """Отметка «клиент жив» для мертвеца: метка в памяти + touch last_poll."""
    STATE.last_client_ts = time.time()
    if DEADMAN:
        try:
            DEADMAN_TOUCH_FILE.touch()
        except OSError:
            pass


def _deadman_destroy(instance_id: int, api_key: str) -> bool:
    """DELETE инстанса через Vast API (ограниченный CONTAINER_API_KEY), 3 попытки."""
    url = "https://console.vast.ai/api/v0/instances/%d/" % instance_id
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(
                url, method="DELETE",
                headers={"Authorization": "Bearer %s" % api_key,
                         "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = int(resp.status)
                log("Мертвец: DELETE %s -> HTTP %d" % (url, status))
                if 200 <= status < 300:
                    return True
        except Exception as e:
            log("WARNING: мертвец: попытка %d не удалась (%s: %s)"
                % (attempt, type(e).__name__, e))
        time.sleep(5)
    return False


def deadman_worker(instance_id: int, api_key: str) -> None:
    """Фоновый поток сторожа: раз в DEADMAN_PERIOD_S секунд — deadman_decision.

    До первого обращения клиента idle считается от старта сервера
    (last_client_ts инициализирован временем старта процесса)."""
    while True:
        time.sleep(DEADMAN_PERIOD_S)
        now = time.time()
        reason = deadman_decision(now - STATE.started_at,
                                  now - STATE.last_client_ts,
                                  DEADMAN_MAX_HOURS, DEADMAN_IDLE_MIN)
        if reason is None:
            continue
        log("МЕРТВЕЦ: САМОУНИЧТОЖЕНИЕ инстанса %d — %s" % (instance_id, reason))
        if _deadman_destroy(instance_id, api_key):
            log("Мертвец: запрос принят — Vast остановит инстанс")
            return
        log("WARNING: мертвец: уничтожение не удалось — повтор через %d с"
            % DEADMAN_PERIOD_S)


def start_deadman() -> None:
    """Активация сторожа (DEADMAN=1): нужны CONTAINER_API_KEY и id инстанса."""
    if not DEADMAN:
        return
    api_key = (os.environ.get("CONTAINER_API_KEY") or "").strip()
    instance_id = parse_instance_id(os.environ)
    if not api_key or instance_id is None:
        log("WARNING: DEADMAN=1, но CONTAINER_API_KEY/id инстанса не найдены — "
            "сторож НЕ активирован")
        return
    threading.Thread(target=deadman_worker, args=(instance_id, api_key),
                     daemon=True).start()
    log("Мертвец: сторож активен (инстанс %d, max %.1f ч, idle %.0f мин)"
        % (instance_id, DEADMAN_MAX_HOURS, DEADMAN_IDLE_MIN))


# --------------------------------------------------------- хранилище заданий --

from collections import OrderedDict


class JobStore:
    """Задания озвучки в памяти. Однопоточный доступ из event loop — без локов.

    Статусы: queued -> processing -> done | error; queued/processing -> cancelled
    (DELETE /job/{id} — клиент передал чанк другому серверу).
    Вытеснение: при превышении cap удаляются старейшие завершённые
    (done/error/cancelled); активные (queued/processing) не вытесняются никогда.
    """

    def __init__(self, cap: int = JOBS_CAP):
        self.cap = cap
        self.jobs: OrderedDict = OrderedDict()  # id -> dict

    def get(self, job_id: str):
        return self.jobs.get(job_id)

    def put_new(self, job_id: str, text: str, language: str) -> dict:
        job = {"status": "queued", "text": text, "language": language,
               "data": None, "error": None, "created": time.time()}
        self.jobs[job_id] = job
        self.jobs.move_to_end(job_id)
        self._evict()
        return job

    def _evict(self) -> None:
        if len(self.jobs) <= self.cap:
            return
        for jid in list(self.jobs.keys()):
            if len(self.jobs) <= self.cap:
                break
            if self.jobs[jid]["status"] in ("done", "error", "cancelled"):
                del self.jobs[jid]

    def counts(self) -> dict:
        c = {"queued": 0, "processing": 0, "done": 0, "error": 0, "cancelled": 0}
        for j in self.jobs.values():
            c[j["status"]] = c.get(j["status"], 0) + 1
        return c


JOB_STORE = JobStore()


# ---------------------- отмена заданий (DELETE /job/{id}, чистые решения) --

def cancel_job_decision(status):
    """Решение DELETE /job/{id} по текущему статусу задания (чистая, --selftest).

    status: None (задание неизвестно) | queued | processing | done | error |
    cancelled. Возврат (http_code, new_status | None):
      None                -> (404, None)        — неизвестный id;
      queued/processing   -> (200, "cancelled") — queued воркеру не отдаётся,
                             у processing прервать генерацию нельзя, но результат
                             не сохранится (см. job_should_run/job_should_store);
      done/error/cancelled -> (200, None)       — идемпотентно, статус не трогаем.
    """
    if status is None:
        return 404, None
    if status in ("queued", "processing"):
        return 200, "cancelled"
    return 200, None


def job_should_run(status) -> bool:
    """Отдавать ли задание воркеру (run_job): только queued. Отменённое
    (cancelled) — не отдаётся: клиент уже передал чанк другому серверу."""
    return status == "queued"


def job_should_store(status) -> bool:
    """Сохранять ли готовый результат генерации: только если задание всё ещё
    processing. DELETE во время генерации переводит его в cancelled — результат
    выбрасывается (сама генерация не прерывается, это невозможно)."""
    return status == "processing"


async def run_job(job_id: str) -> None:
    """Фоновая корутина одного задания: разметка ударений (опц.) -> движок ->
    ASR-QC (опц.) -> WAV-байты контракта -> done/error.

    vllm: прямой HTTP-запрос к локальному vLLM (семафор VLLM_CONCURRENCY);
    inprocess: динамический Batcher (путь v2). ASR-QC сравнивает транскрипт
    с ЧИСТЫМ текстом job["text"] — разметка живёт только в tts_text.

    Отмена (DELETE /job/{id}): cancelled-задание воркеру не отдаётся
    (job_should_run); отмена во время генерации — результат не сохраняется
    (job_should_store), исход задания остаётся cancelled.
    """
    job = JOB_STORE.get(job_id)
    if job is None or not job_should_run(job["status"]):
        return  # отменено (или уже не queued) — воркеру не отдаём
    job["status"] = "processing"
    try:
        clean_text = job["text"]
        tts_text = await prepare_tts_text(clean_text)
        data = await synth_with_qc(job_id, clean_text, tts_text,
                                   job["language"])
        if not job_should_store(job["status"]):
            log("Задание %s отменено во время генерации — результат выброшен"
                % job_id)
            return
        job["data"] = data
        job["status"] = "done"
    except Exception as e:
        if not job_should_store(job["status"]):
            # Отменённое задание: его ошибка никому не нужна, статус не трогаем
            log("Задание %s отменено; ошибка генерации проигнорирована (%s: %s)"
                % (job_id, type(e).__name__, e))
            return
        job["error"] = "%s: %s" % (type(e).__name__, e)
        job["status"] = "error"
        log("ОШИБКА генерации %s: %s" % (job_id, job["error"]))


# ------------------------------------------------------------------- FastAPI --

class TTSRequest(BaseModel):
    id: str
    text: str
    language: str = "Russian"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Батчер (движок inprocess) — в event loop; семафор vllm — здесь же;
    # тяжёлый старт — в фоновом потоке (ready=False пока идёт)
    STATE.batcher = Batcher(generate_batch)
    STATE.vllm_sem = asyncio.Semaphore(VLLM_CONCURRENCY)
    batcher_task = asyncio.create_task(STATE.batcher.run())
    threading.Thread(target=startup_worker, daemon=True).start()
    # Сторож-«мертвец» — независимо от исхода старта (упавший старт тоже жжёт
    # деньги; idle-отсчёт до первого клиента идёт от старта сервера)
    start_deadman()
    yield
    await STATE.batcher.stop()
    try:
        await asyncio.wait_for(batcher_task, timeout=5)
    except Exception:
        batcher_task.cancel()
    if STATE.vllm_client is not None:
        try:
            await STATE.vllm_client.aclose()
        except Exception:
            pass
    _kill_vllm()  # сабпроцесс vLLM не должен переживать сервер


app = FastAPI(title="AudiobookStudio TTS", lifespan=lifespan)


def _check_auth(authorization) -> None:
    """401 — неверный/отсутствующий токен (сравнение без утечки по времени)."""
    expected = "Bearer %s" % AUTH_TOKEN
    if not AUTH_TOKEN or not secrets.compare_digest(
            (authorization or "").encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="неверный токен")


@app.get("/health")
async def health():
    """Статус сервера — без авторизации (диспетчер поллит готовность).

    Поле engine (v3): null пока идёт старт, затем "vllm" | "inprocess".
    queue: vllm — задания queued+processing; inprocess — очередь батчера (как в v2).
    """
    jobs = JOB_STORE.counts()
    if STATE.engine == "vllm":
        queue = jobs["queued"] + jobs["processing"]
    else:
        queue = STATE.batcher.pending() if STATE.batcher else 0
    return {
        "status": "ok",
        "ready": STATE.ready,
        "model": MODEL_ID,
        "engine": STATE.engine,
        "queue": queue,
        "jobs": jobs,
        "error": STATE.error,
    }


@app.post("/tts")
async def tts(req: TTSRequest, authorization: str = Header(default=None)):
    """Принять задание озвучки. Мгновенный ответ, аудио забирается GET /result/{id}.

    Идемпотентен: повтор с тем же id вернёт текущий статус, не создавая дубля.
    Повтор для error, отменённого (cancelled) или уже вытесненного done —
    перезапускает задание.
    """
    _check_auth(authorization)
    _note_client_activity()  # мертвец: клиент жив
    # 503 — модель ещё грузится (или старт упал: ошибка видна в /health)
    if not STATE.ready:
        raise HTTPException(status_code=503,
                            detail="сервер не готов: %s" % (STATE.error or "идёт старт"))
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="пустой текст")

    job = JOB_STORE.get(req.id)
    if job is not None:
        if job["status"] in ("queued", "processing"):
            return {"status": job["status"]}
        if job["status"] == "done" and job["data"] is not None:
            return {"status": "done"}
        # error/cancelled либо done без данных (вытеснены) — перезапуск задания
    JOB_STORE.put_new(req.id, text, req.language)
    asyncio.get_running_loop().create_task(run_job(req.id))
    return {"status": "queued"}


@app.get("/result/{chunk_id}")
async def result(chunk_id: str, authorization: str = Header(default=None)):
    """Забрать результат задания: 404/202/500/200+WAV (см. контракт в шапке)."""
    _check_auth(authorization)
    _note_client_activity()  # мертвец: клиент жив
    job = JOB_STORE.get(chunk_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задание неизвестно")
    if job["status"] == "cancelled":  # повторный POST /tts перезапустит задание
        raise HTTPException(status_code=404, detail="задание отменено (DELETE /job)")
    if job["status"] in ("queued", "processing"):
        return JSONResponse(status_code=202, content={"status": job["status"]})
    if job["status"] == "error":
        return JSONResponse(status_code=500, content={"error": job["error"]})
    if job["data"] is None:  # done, но данные вытеснены — нужен повторный POST
        raise HTTPException(status_code=404, detail="результат вытеснен, повтори POST /tts")
    return Response(content=job["data"], media_type="audio/wav",
                    headers={"X-Chunk-Id": chunk_id})


@app.delete("/job/{chunk_id}")
async def cancel_job(chunk_id: str, authorization: str = Header(default=None)):
    """Отменить устаревшее задание (клиент передал чанк другому серверу).

    Решение — cancel_job_decision (см. её докстринг): queued/processing ->
    cancelled, терминальные -> 200 без изменений, неизвестный id -> 404.
    Идемпотентен: повторный DELETE того же id безопасен.
    """
    _check_auth(authorization)
    _note_client_activity()  # мертвец: клиент жив
    job = JOB_STORE.get(chunk_id)
    old_status = None if job is None else job["status"]
    code, new_status = cancel_job_decision(old_status)
    if code == 404:
        raise HTTPException(status_code=404, detail="задание неизвестно")
    if new_status is not None:
        job["status"] = new_status
        job["data"] = None  # результат отменённого не нужен — не держим память
        log("Задание %s отменено клиентом (был статус %s)"
            % (chunk_id, old_status))
    return {"status": job["status"]}


# ------------------------------------------------------------------ selftest --

def _selftest_window() -> None:
    """Выбор 15-секундного окна из фейковых word-timestamps."""
    words = []
    # редкие слова: 0-18 c, каждые 2 c (10 слов)
    for k in range(10):
        t = 2.0 * k
        words.append({"start": t, "end": t + 0.3, "word": "редкое%d" % k})
    # плотный участок: 20-30 c, каждые 0.25 c (40 слов)
    for k in range(40):
        t = 20.0 + 0.25 * k
        words.append({"start": t, "end": t + 0.2, "word": "плотное%d" % k})
    # снова редкие: 32-58 c, каждые 2 c (14 слов)
    for k in range(14):
        t = 32.0 + 2.0 * k
        words.append({"start": t, "end": t + 0.3, "word": "хвост%d" % k})

    t0, t1, text = pick_ref_window(words, 15.0)
    n_in = len(text.split())
    assert t1 - t0 <= 15.0 + 1e-9, "окно длиннее 15 c: %.3f" % (t1 - t0)
    assert t0 <= 20.0 and t1 >= 29.95, "окно (%.2f-%.2f) не накрыло плотный участок" % (t0, t1)
    assert n_in >= 40, "в окне %d слов, ожидалось >= 40" % n_in
    assert "плотное0" in text and "плотное39" in text

    # короткий сэмпл: все слова помещаются в окно целиком
    short = [{"start": i * 1.0, "end": i * 1.0 + 0.4, "word": "с%d" % i} for i in range(5)]
    t0s, t1s, ts = pick_ref_window(short, 15.0)
    assert ts.split() == ["с0", "с1", "с2", "с3", "с4"]
    assert t0s == 0.0 and abs(t1s - 4.4) < 1e-9

    # пустой список — ошибка
    try:
        pick_ref_window([], 15.0)
        raise AssertionError("пустой список должен давать ValueError")
    except ValueError:
        pass
    print("selftest 1/13: выбор 15-секундного окна (words) — OK "
          "(окно %.2f-%.2f c, %d слов)" % (t0, t1, n_in))


def _selftest_lively_window() -> None:
    """«Живое» окно: чередование тихо/громко должно победить монотонные участки."""
    sr = SAMPLE_RATE

    def block(seconds: float, amp: float):
        # постоянная амплитуда: RMS каждого фрейма == amp (детерминированно)
        return np.full(int(seconds * sr), amp, dtype=np.float64)

    # 0-20 c: монотонно громко (дисперсия RMS ~0);
    # 20-40 c: живой участок — чередование 1 c тихо (0.05) / 1 c громко (0.9);
    # 40-60 c: монотонно тихо.
    parts = [block(20.0, 0.6)]
    for _ in range(10):
        parts.append(block(1.0, 0.05))
        parts.append(block(1.0, 0.9))
    parts.append(block(20.0, 0.2))
    x = np.concatenate(parts)

    t0, t1 = pick_ref_window_lively(x, sr, 15.0)
    assert abs((t1 - t0) - 15.0) < REF_RMS_FRAME_S + 1e-9, \
        "окно не ~15 c: %.3f" % (t1 - t0)
    assert 19.5 <= t0 and t1 <= 40.5, \
        "окно (%.2f-%.2f) не накрыло живой участок 20-40 c" % (t0, t1)

    # запись короче окна — берётся целиком
    t0s, t1s = pick_ref_window_lively(block(5.0, 0.3), sr, 15.0)
    assert t0s == 0.0 and abs(t1s - 5.0) <= REF_RMS_FRAME_S + 1e-9

    # пустой массив — ошибка
    try:
        pick_ref_window_lively(np.array([]), sr, 15.0)
        raise AssertionError("пустой массив должен давать ValueError")
    except ValueError:
        pass
    print("selftest 2/13: «живое» окно (макс. дисперсия RMS, тихо/громко побеждает "
          "монотон) — OK (окно %.2f-%.2f c)" % (t0, t1))


def _selftest_wav() -> None:
    """Сборка WAV-байтов из фейкового numpy-массива (24000 Гц, mono, s16, RIFF)."""
    arr = np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32)  # 2.0 -> клип
    data = wav_bytes_from_array(arr, SAMPLE_RATE)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "нет RIFF/WAVE заголовка"
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getnchannels() == 1, "не mono"
        assert w.getsampwidth() == 2, "не s16"
        assert w.getframerate() == SAMPLE_RATE, "частота != 24000"
        assert w.getnframes() == 5
        decoded = np.frombuffer(w.readframes(5), dtype=np.int16)
    expected = np.array([0, 16383, -16383, 32767, -32767], dtype=np.int16)
    assert np.array_equal(decoded, expected), "значения: %s != %s" % (decoded, expected)

    # int16-массив проходит без масштабирования
    raw = np.array([-32768, 0, 32767], dtype=np.int16)
    data2 = wav_bytes_from_array(raw, SAMPLE_RATE)
    with wave.open(io.BytesIO(data2), "rb") as w:
        decoded2 = np.frombuffer(w.readframes(3), dtype=np.int16)
    assert np.array_equal(decoded2, raw)
    print("selftest 3/13: WAV-байты (RIFF, 24000 Гц, mono, s16, клиппинг) — OK "
          "(%d байт)" % len(data))


def _selftest_batcher() -> None:
    """Батчер на фейковой модели: 8+3 запроса должны собраться батчами [8, 3]."""
    calls = []  # размеры собранных батчей

    def fake_generate(texts, langs):
        # имитация блокирующей GPU-генерации в ThreadPoolExecutor(1)
        calls.append(len(texts))
        assert all(lang == "Russian" for lang in langs)
        time.sleep(0.2)
        # фейковый wav: значение сэмплов кодирует номер запроса
        wavs = [np.full(8, float(t.split("_")[1]) / 100.0, dtype=np.float32)
                for t in texts]
        return wavs, SAMPLE_RATE

    async def scenario():
        b = Batcher(fake_generate, batch_size=8, window_s=0.15)
        runner = asyncio.create_task(b.run())
        first = [asyncio.create_task(b.submit("req_%d" % i, "Russian"))
                 for i in range(8)]
        await asyncio.sleep(0.05)  # 3 «опоздавших» прилетают во время генерации
        second = [asyncio.create_task(b.submit("req_%d" % i, "Russian"))
                  for i in range(8, 11)]
        results = await asyncio.gather(*(first + second))
        await b.stop()
        await runner
        return results

    results = asyncio.run(scenario())
    assert calls == [8, 3], "батчи собрались как %s, ожидалось [8, 3]" % calls
    for i, (samples, sr) in enumerate(results):
        assert sr == SAMPLE_RATE
        assert abs(float(samples[0]) - i / 100.0) < 1e-6, \
            "запрос %d получил чужой результат" % i
    print("selftest 4/13: батчер (asyncio, 11 запросов -> батчи %s, "
          "результаты по адресатам) — OK" % calls)


def _selftest_engine_choice() -> None:
    """Логика выбора движка: auto с недоступным vllm -> inprocess, и остальные ветки."""
    calls = []

    def ok_starter():
        calls.append("ok")

    def broken_starter():
        raise RuntimeError("vllm недоступен (фейк selftest)")

    def forbidden_starter():
        raise AssertionError("starter не должен вызываться при TTS_ENGINE=inprocess")

    # главный сценарий фолбэка: auto + сломанный vllm -> inprocess (WARNING в лог)
    assert choose_engine("auto", broken_starter) == "inprocess"
    # auto + рабочий vllm -> vllm (starter вызван ровно один раз)
    assert choose_engine("auto", ok_starter) == "vllm" and calls == ["ok"]
    # явный inprocess: starter вообще не трогаем
    assert choose_engine("inprocess", forbidden_starter) == "inprocess"
    # регистр не важен
    assert choose_engine("VLLM", ok_starter) == "vllm"
    # явный vllm со сломанным стартом обязан падать (фолбэка нет)
    try:
        choose_engine("vllm", broken_starter)
        raise AssertionError("явный vllm с ошибкой старта должен пробрасывать исключение")
    except RuntimeError:
        pass
    # неизвестное значение трактуется как auto (WARNING + фолбэк)
    assert choose_engine("абракадабра", broken_starter) == "inprocess"
    print("selftest 5/13: выбор движка (auto->фолбэк inprocess, vllm->фатал, "
          "inprocess->без vllm) — OK")


def _selftest_vllm_payload() -> None:
    """Формат запроса POST /v1/audio/speech к vLLM — оба режима голоса."""
    p1 = build_vllm_payload("Привет, мир.", "Russian", "named",
                            "/workspace/ref.wav", "эталонный текст")
    assert p1 == {"model": MODEL_ID, "input": "Привет, мир.",
                  "language": "Russian", "voice": VLLM_VOICE_NAME}, p1
    p2 = build_vllm_payload("Привет, мир.", "Russian", "per_request",
                            "/workspace/ref.wav", "эталонный текст")
    from pathlib import Path as _P
    exp_uri = _P("/workspace/ref.wav").absolute().as_uri()  # file://... (vLLM-omni требует URI)
    assert p2 == {"model": MODEL_ID, "input": "Привет, мир.",
                  "language": "Russian", "ref_audio": exp_uri,
                  "ref_text": "эталонный текст"}, p2
    assert p2["ref_audio"].startswith("file://"), p2
    # temperature (ретраи ASR-QC): появляется в payload только когда задана
    p3 = build_vllm_payload("Т.", "Russian", "named", temperature=0.945)
    assert p3["temperature"] == 0.945 and "temperature" not in p1, p3
    print("selftest 6/13: payload vLLM (named: voice=%s; per_request: "
          "ref_audio+ref_text; temperature опциональна) — OK" % VLLM_VOICE_NAME)


def _selftest_transcode() -> None:
    """Детекция контракта WAV 24k mono s16 и перекодировка ffmpeg-ом чужого формата."""
    # контрактный WAV проходит без изменений (тот же объект — без копий)
    good = wav_bytes_from_array(np.zeros(2400, dtype=np.int16))
    assert wav_matches_contract(good), "контрактный WAV не распознан"
    assert ensure_wav_contract(good) is good, "контрактный WAV не должен перекодироваться"
    # мусор и пустота — не контракт
    assert not wav_matches_contract(b"")
    assert not wav_matches_contract(b"\x00" * 64)

    # 48 кГц стерео — валидный WAV, но не контракт
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        t = np.arange(4800, dtype=np.float64)  # 0.1 c
        tone = (np.sin(2 * np.pi * 440.0 * t / 48000.0) * 12000).astype(np.int16)
        w.writeframes(np.column_stack([tone, tone]).tobytes())
    bad = buf.getvalue()
    assert not wav_matches_contract(bad), "48k stereo ошибочно принят за контракт"

    fb = ffmpeg_bin()
    if not (Path(fb).exists() or shutil.which(fb)):
        print("selftest 7/13: перекодировка — ffmpeg не найден, проверена только "
              "детекция формата (OK)")
        return
    out = ensure_wav_contract(bad)
    assert wav_matches_contract(out), "после перекодировки не контракт"
    with wave.open(io.BytesIO(out), "rb") as w:
        frames = w.getnframes()
    # 0.1 c при 24 кГц ~ 2400 фреймов (ресемплер может чуть сдвинуть края)
    assert 2000 <= frames <= 2900, "длительность после перекодировки: %d фреймов" % frames
    print("selftest 7/13: перекодировка ffmpeg 48k stereo -> 24k mono s16 — OK "
          "(%d фреймов)" % frames)


def _selftest_stress_convert() -> None:
    """Конвертер «+гласная» -> акут ПОСЛЕ гласной + фильтр анти-«Оюрдо»."""
    A = ACUTE
    # несколько слов, ударение в середине и в конце
    assert convert_plus_to_acute("Мен+я зов+ут") == "Меня%s зову%sт" % (A, A)
    assert convert_plus_to_acute("окн+о") == "окно" + A
    # анти-«Оюрдо»: «О»/«о» в НАЧАЛЕ слова остаётся без акута...
    assert convert_plus_to_acute("+Ордо") == "Ордо"
    assert convert_plus_to_acute("+осень пришла") == "осень пришла"
    assert convert_plus_to_acute("«+Ордо»") == "«Ордо»"  # пунктуация — не буква слова
    # ...а «о» в СЕРЕДИНЕ/конце слова ударение сохраняет (фильтр только для начала)
    assert convert_plus_to_acute("окн+о и +Ордо") == "окно%s и Ордо" % A
    # не-«О» в начале слова размечается как обычно
    assert convert_plus_to_acute("+ужин") == "у%sжин" % A
    # «+» не перед русской гласной — не разметка, остаётся как есть
    assert convert_plus_to_acute("2+2=4") == "2+2=4"
    assert convert_plus_to_acute("C++ и +Ордо") == "C++ и Ордо"
    print("selftest 8/13: конвертер «+»->акут и анти-«Оюрдо» — OK")


def _selftest_stress_dict() -> None:
    """Словарь ударений: разбор файла, замены, приоритет над акцентором."""
    A = ACUTE
    # разбор формата stress_dict.yaml (построчный парсер работает и без PyYAML)
    raw = ("# шапка-комментарий: не запись\n"
           "\n"
           "Джинн: \"Джи" + A + "нн\"\n"
           "замок: 'за" + A + "мок'  # инлайн-комментарий\n")
    d = parse_stress_dict(raw)
    assert d == {"Джинн": "Джи" + A + "нн", "замок": "за" + A + "мок"}, d

    # замена слова целиком, регистронезависимо; замена — ровно как в словаре
    out = apply_stress_dict("джинн вышел из замок", d)
    assert out == "Джи%sнн вышел из за%sмок" % (A, A), out
    # слово, уже размеченное клиентом, словарь не трогает
    pre = "Джи" + A + "нн"
    assert apply_stress_dict(pre, {"Джинн": "ДЖИНН"}) == pre

    # словарь выигрывает у акцентора; остальное размечает акцентор
    def fake_accentor(text):
        assert ACUTE not in text, "акцентору должен идти текст без акутов"
        return text.replace("замок", "зам+ок").replace("стоит", "сто+ит")

    out2 = mark_stress("замок стоит", fake_accentor, d)
    assert out2 == "за%sмок стои%sт" % (A, A), out2  # НЕ «замо́к» акцентора
    # клиентская разметка в тексте тоже выигрывает (без словаря)
    out3 = mark_stress("Джи" + A + "нн стоит", fake_accentor, {})
    assert out3 == "Джи%sнн стои%sт" % (A, A), out3
    # без акцентора работает только словарь; без всего — текст как есть
    assert mark_stress("замок", None, d) == "за" + A + "мок"
    assert mark_stress("привет", None, {}) == "привет"
    print("selftest 9/13: словарь ударений (разбор, замены, приоритет над "
          "акцентором) — OK")


def _selftest_asr_normalize() -> None:
    """Нормализация текста перед CER: регистр, пунктуация, ё->е, акуты, цифры."""
    assert normalize_for_asr("Привет,   МИР!!!") == "привет мир"
    assert normalize_for_asr("Ёжик — в тумане... (да)") == "ежик в тумане да"
    assert normalize_for_asr("Съешь ЕЩЁ этих мягких булок") == \
        "съешь еще этих мягких булок"
    # акуты разметки не должны влиять на сравнение
    assert normalize_for_asr("за" + ACUTE + "мок") == "замок"
    assert normalize_for_asr("") == ""
    # цифры транскрипта -> слова (если есть num2words; иначе остаются цифрами)
    try:
        import num2words  # noqa: F401
        assert normalize_for_asr("им 5 лет") == "им пять лет"
        note = "num2words есть: «5»->«пять»"
    except ImportError:
        assert normalize_for_asr("им 5 лет") == "им 5 лет"
        note = "num2words нет: цифры остаются"
    print("selftest 10/13: нормализация ASR-текста — OK (%s)" % note)


def _selftest_cer_qc() -> None:
    """CER: пары строк с известным расстоянием + лестница температур ретраев."""
    assert levenshtein("", "") == 0
    assert levenshtein("абв", "") == 3 and levenshtein("", "абв") == 3
    assert levenshtein("кот", "кот") == 0
    assert levenshtein("кот", "кто") == 2        # две замены (транспозиций нет)
    assert levenshtein("караван", "карман") == 2  # замена + удаление
    assert levenshtein("привет", "привед") == 1
    assert cer("привет", "привет") == 0.0
    assert abs(cer("привет", "привед") - 1.0 / 6.0) < 1e-12
    assert cer("абв", "") == 1.0
    # длинная пара (порядок чанка ~1500 симв) — векторизованный путь
    a = "абвгд " * 250
    b = a.replace("в", "ф", 3)
    assert levenshtein(a, b) == 3
    assert abs(cer(a, b) - 3.0 / len(a)) < 1e-12
    # лестница ретраев: чуть теплее -> чуть холоднее
    assert qc_retry_temps(1.0, 2) == [1.05, 0.95]
    assert qc_retry_temps(0.8, 1) == [round(0.8 * 1.05, 4)]
    assert qc_retry_temps(0.9, 0) == []
    print("selftest 11/13: CER (Левенштейн, полоса numpy) и лестница ретраев "
          "ASR-QC — OK")


def _selftest_deadman() -> None:
    """Решение мертвеца (dry-run от подставных времён) + парсинг id инстанса."""
    # жить: аптайм и молчание в норме
    assert deadman_decision(3600, 60, 4.5, 45) is None
    # аптайм превышен -> причина
    r = deadman_decision(4.6 * 3600, 0, 4.5, 45)
    assert r is not None and "аптайм" in r, r
    # клиент молчит дольше лимита -> причина
    r2 = deadman_decision(600, 46 * 60, 4.5, 45)
    assert r2 is not None and "молчит" in r2, r2
    # ровно на лимите — ещё живём (строгое «больше»)
    assert deadman_decision(4.5 * 3600, 45 * 60, 4.5, 45) is None
    # id инстанса из env контейнера
    assert parse_instance_id({"CONTAINER_ID": "123456"}) == 123456
    assert parse_instance_id({"VAST_CONTAINERLABEL": "C.98765"}) == 98765
    assert parse_instance_id({"CONTAINER_ID": "", "VAST_CONTAINERLABEL": "C.5"}) == 5
    assert parse_instance_id({}) is None
    print("selftest 12/13: решение мертвеца (uptime/idle, dry-run) и парсинг "
          "id инстанса — OK")


def _selftest_cancel() -> None:
    """Отмена заданий (DELETE /job/{id}): решения — чистые функции, без GPU."""
    # Решение DELETE по статусу: неизвестное -> 404; активные -> cancelled;
    # терминальные -> 200 без изменений (идемпотентность)
    assert cancel_job_decision(None) == (404, None)
    assert cancel_job_decision("queued") == (200, "cancelled")
    assert cancel_job_decision("processing") == (200, "cancelled")
    assert cancel_job_decision("done") == (200, None)
    assert cancel_job_decision("error") == (200, None)
    assert cancel_job_decision("cancelled") == (200, None)  # повторный DELETE

    # Отдавать ли воркеру: только queued (cancelled — «убрано из очереди»)
    assert job_should_run("queued")
    assert not job_should_run("cancelled")
    assert not job_should_run("processing")
    assert not job_should_run("done")

    # Сохранять ли результат: только processing (DELETE во время генерации
    # переводит в cancelled — готовый WAV выбрасывается)
    assert job_should_store("processing")
    assert not job_should_store("cancelled")
    assert not job_should_store("done") and not job_should_store("error")

    # Вытеснение: cancelled вытесняется как done/error, активные — никогда
    js = JobStore(cap=2)
    js.put_new("a", "т", "Russian")["status"] = "cancelled"
    js.put_new("b", "т", "Russian")            # queued — не вытесняется
    js.put_new("c", "т", "Russian")            # cap превышен -> уходит "a"
    assert "a" not in js.jobs and "b" in js.jobs and "c" in js.jobs, \
        "вытесниться должно было cancelled-задание 'a'"
    counts = js.counts()
    assert counts["cancelled"] == 0 and counts["queued"] == 2, counts
    print("selftest 13/13: отмена заданий (решение DELETE, воркер, сохранение "
          "результата, вытеснение cancelled) — OK")


def run_selftest() -> int:
    """Чистая логика без GPU (и без silero/faster-whisper/qwen): окна, WAV, батчер,
    выбор движка, payload, перекодировка, разметка ударений, ASR-нормализация,
    CER, решение мертвеца, отмена заданий."""
    print("=== server.py --selftest (без GPU) ===")
    try:
        _selftest_window()
        _selftest_lively_window()
        _selftest_wav()
        _selftest_batcher()
        _selftest_engine_choice()
        _selftest_vllm_payload()
        _selftest_transcode()
        _selftest_stress_convert()
        _selftest_stress_dict()
        _selftest_asr_normalize()
        _selftest_cer_qc()
        _selftest_deadman()
        _selftest_cancel()
    except Exception:
        traceback.print_exc()
        print("SELFTEST FAILED")
        return 1
    print("SELFTEST OK")
    return 0


# --------------------------------------------------------------------- main --

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    if not AUTH_TOKEN:
        print("ОШИБКА: env AUTH_TOKEN обязателен для запуска сервера", flush=True)
        sys.exit(2)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
