# HW3 Runbook — Speculative Decoding & Quantization

Пайплайн ускорения `Qwen/Qwen3-8B` на **1× H100 80GB** (нужно ~140GB диска).
Все команды запускаются на удалённом H100. Локальная RTX 3060 задачу не тянет.

> Каждый скрипт имеет `--help`. Флаги CLI Speculators (`generate-data`,
> `train eagle3`) и vLLM (`--speculative-config`) закреплены под версии из плана
> (speculators `v0.5.0`, vLLM `0.20.0`). Если установленный тег отличается —
> сверьтесь с `speculators --help` и `vllm serve --help` и поправьте обёртки.

## Три окружения (dependency-конфликт → раздельно)

| venv | Назначение | Ключевые пакеты |
| --- | --- | --- |
| `speculators_venv` | данные, hidden states, обучение | `speculators==v0.5.0` (editable) |
| `vllm_venv` | serve + benchmark | `vllm==0.20.0`, `fastapi<0.137` |
| `comp_venv` | FP8-квантизация | `llmcompressor==0.12.0` |

```bash
bash env_setup.sh          # создаёт все 3 venv, клонирует speculators, кеширует модель
export HF_HOME="$PWD/../.hf_home"   # общий кеш модели во всех шеллах
```

## Задача 1 — Окружение и данные

Обёртки вызывают реальные скрипты из склонированного репозитория speculators
(`speculators/scripts/*.py`). Ключевое: сервер извлечения hidden states — это
**vLLM** (`launch_vllm.py` → `python -m vllm ... --speculative_config
extract_hidden_states`), он живёт в **`vllm_venv`**; а `prepare_data.py` и
`data_generation_offline.py` (openai-клиент по HTTP) — в **`speculators_venv`**.
Два процесса общаются по HTTP (`--endpoint`). vLLM 0.20.0 подходит (нужно ≥0.18).

```bash
# 1) Скачать + отфильтровать ShareGPT в .jsonl (формат совместим со speculators)
source speculators_venv/bin/activate
python prepare_data.py --model Qwen/Qwen3-8B --max-samples 3000 --seq-len 2048 \
       --out data/sharegpt_qwen3.jsonl

# 2) preprocess + hidden states. По умолчанию сервер поднимается автоматически
#    через ./vllm_venv/bin/python; hidden states -> output/hidden_states
python generate_hidden_states.py --model Qwen/Qwen3-8B \
       --data data/sharegpt_qwen3.jsonl --work-dir output \
       --seq-length 2048 --max-samples 3000 --concurrency 32 --min-free-gb 20
deactivate
```

Вариант с двумя терминалами (эквивалент, если авто-запуск не подходит):

```bash
# Терминал A (vllm_venv): сервер извлечения hidden states
source vllm_venv/bin/activate
python launch_vllm.py Qwen/Qwen3-8B --port 8000

# Терминал B (speculators_venv): preprocess + генерация против готового сервера
source speculators_venv/bin/activate
python generate_hidden_states.py --model Qwen/Qwen3-8B \
       --data data/sharegpt_qwen3.jsonl --work-dir output \
       --seq-length 2048 --max-samples 3000 --no-manage-server \
       --endpoint http://localhost:8000/v1
```

Следите за `df -h` — hidden states ≈ ~140GB. При нехватке диска сначала снижайте
`--max-samples`. Ошибка «missing temporary file» → скрипт сам чистит
`/tmp/hidden_states/*`. Несовпадение seq-len ловит `--validate-outputs`; если
падает — проверьте версию vLLM.

## Задача 2 — Обучение draft-головы EAGLE-3

```bash
source speculators_venv/bin/activate
python train_eagle3.py --verifier Qwen/Qwen3-8B \
       --data-path output --hidden-states output/hidden_states \
       --save-path output/checkpoints --epochs 5 --total-seq-len 2048 \
       --on-missing raise --logger tensorboard
deactivate
```

Обёртка запускает `torchrun scripts/train.py`. Чекпоинты — под
`output/checkpoints/` (лучший сохраняет сам тренер, обычно `best/`). Ориентир:
`full_acc_0 ≈ 0.46`, падает по позициям (метрики — в `--log-dir logs`). Низкий
`full_acc_0` → чинить Задачу 1.

## Задача 3 — FP8 dynamic квантизация

```bash
source comp_venv/bin/activate
python quantize_fp8.py --model Qwen/Qwen3-8B --out Qwen3-8B-FP8-Dynamic
deactivate
```

`Linear` → FP8, `lm_head` не трогаем, база BF16 не перезаписывается. Скрипт
проверяет секцию quantization в `config.json`.

## Задача 4 — Serve + benchmark (4 конфигурации)

Единые настройки для всех: mt-bench, 80 промптов, concurrency 8, seed 0, prefix
caching выкл. Сервер (`serve.sh`) и бенч (`run_benchmark.sh`) — в разных шеллах;
либо используйте `sweep_draft_tokens.py`, который поднимает/гасит сервер сам.

```bash
source vllm_venv/bin/activate

# 1) Baseline BF16
bash serve.sh --model Qwen/Qwen3-8B                       # шелл A
bash run_benchmark.sh Qwen/Qwen3-8B baseline              # шелл B

# 2) Spec-dec (BF16 + EAGLE-3) — тюнинг числа draft-токенов
python sweep_draft_tokens.py --model Qwen/Qwen3-8B \
       --draft-head output/checkpoints/best --values 1 2 3 4 --label-prefix spec

# 3) FP8
bash serve.sh --model Qwen3-8B-FP8-Dynamic               # шелл A
bash run_benchmark.sh Qwen3-8B-FP8-Dynamic fp8           # шелл B

# 4) FP8 + Spec-dec — тюнинг заново (значение НЕ переносить из п.2)
python sweep_draft_tokens.py --model Qwen3-8B-FP8-Dynamic \
       --draft-head output/checkpoints/best --values 1 2 3 --label-prefix fp8_spec
```

Результаты бенчей сохраняются в `results/*.txt` — вставьте блоки
`Serving Benchmark Result` в TODO-ячейки ноутбука.

### Пороги оценки (Output token throughput, tok/s)

| Конфигурация | Порог | Баллы |
| --- | ---: | ---: |
| Spec-dec (EAGLE-3) | > 1250 | 25 |
| FP8 dynamic | > 1550 | 10 |
| FP8 + Spec-dec (с тюнингом) | > 1750 | 15 |

Референс: Baseline 841 · Spec-dec 1259 (draft=2) · FP8 1567 ·
FP8+Spec 1767 (draft=1) tok/s.

## Главный вопрос

**Сначала квантизация, затем обучение draft-головы против квантованного
верификатора** — квантизация смещает распределение верификатора, поэтому
acceptance у FP8+spec отличается (референс: 36.5% vs 22.5% у BF16) и оптимальное
число draft-токенов другое. Обоснование — измеренные throughput/acceptance/TPOT.

## Что НЕ коммитить

`*_venv/`, `speculators/`, `data/`, `output/`, `results/`, `Qwen3-8B-FP8-Dynamic/`,
`.hf_home/` — см. `.gitignore`.
