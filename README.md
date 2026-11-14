# Triton Feedback Intelligence

CPU-only ML-система для классификации клиентских обращений
и генерации ответов с использованием NVIDIA Triton Inference Server.

Учебный проект курса OTUS LLM Engineering.

## Цель проекта

Исследовать работу нескольких ML-моделей под управлением NVIDIA
Triton Inference Server, реализовать Business Logic Scripting (BLS)
и измерить производительность системы при конкурентной нагрузке.

Выполнение на CPU согласовано с преподавателем.

## Архитектура

Пользовательский текст
        |
        v
feedback_pipeline (Python BLS)
        |
        v
embedding_model (multilingual-e5-small)
        |
        v
classifier (Logistic Regression / ONNX)
        |
        +---- positive ----> Результат без генерации
        |
        +---- complaint / question
                    |
                    v
           llm_generator (Qwen2.5-0.5B)
                    |
                    v
               JSON-ответ

BLS вызывает модели внутри Triton через pb_utils.InferenceRequest.

## Модели

| Компонент | Backend | Назначение |
|---|---|---|
| embedding_model | Python | Эмбеддинги длиной 384 |
| classifier | ONNX Runtime | Классификация на 3 категории |
| llm_generator | Python + llama.cpp | Генерация текста |
| feedback_pipeline | Python BLS | Оркестрация моделей |

Embedding-модель: intfloat/multilingual-e5-small.

Классификатор: Logistic Regression, обученная на 30
демонстрационных русскоязычных примерах.

LLM: Qwen2.5-0.5B-Instruct, GGUF Q4_K_M.

Для E5 используется префикс "query: " и нормализация векторов.
Эти правила одинаковы при обучении и инференсе.

## Model Repository

model_repository/
├── embedding_model/
│   ├── 1/model.py
│   └── config.pbtxt
├── classifier/
│   ├── 1/model.onnx
│   ├── config.pbtxt
│   └── metadata.json
├── llm_generator/
│   ├── 1/model.py
│   ├── 1/qwen2.5-0.5b-instruct-q4_k_m.gguf
│   └── config.pbtxt
└── feedback_pipeline/
    ├── 1/model.py
    └── config.pbtxt

Веса моделей исключены из Git.
Их необходимо подготовить отдельно перед запуском.

## Инфраструктура

Локальная разработка:
- Ubuntu Linux
- Python 3.12
- uv
- ONNX Runtime CPU
- llama.cpp CPU

Сервер:
- Yandex Cloud Compute Cloud
- Ubuntu 24.04
- 16 vCPU
- 64 GiB RAM
- NVIDIA Triton 2.74.0 (сборка CPU-only)

Порты Triton:
- 8000: HTTP
- 8001: gRPC
- 8002: Prometheus Metrics

Внешний доступ к портам Triton не требуется.
Используется SSH-туннель.

## Подготовка моделей

Локальная установка зависимостей:

    uv venv --python 3.12 .venv
    source .venv/bin/activate
    uv pip install -r requirements.txt

Обучение и экспорт классификатора:

    python scripts/train_classifier.py

Подготовка нагрузочных данных:

    python scripts/prepare_benchmarks.py

Локальный интеграционный тест:

    python scripts/test_local_pipeline.py

## Запуск Triton

Сервер собирается из официальных исходников NVIDIA
с ONNX Runtime и Python Backend, без GPU.

Dockerfile и docker-compose.yml подготовлены.

Окончательные команды развёртывания и процедура установки
зависимостей Python Backend будут добавлены после
успешного интеграционного запуска.

## Проверка

Локально проверены:
- экспорт классификатора в ONNX;
- инференс через ONNX Runtime CPU;
- создание эмбеддингов E5;
- генерация Qwen через llama.cpp;
- последовательная обработка трёх тестовых обращений;
- синтаксис Python-скриптов моделей и BLS;
- подготовка входных данных для нагрузочного тестирования.

Проверка моделей внутри Triton — в процессе.

## Нагрузочное тестирование

Планируется использование NVIDIA Perf Analyzer.

Сценарии:
- отдельная модель классификации;
- отдельная модель эмбеддингов;
- отдельная генеративная модель;
- BLS с генерацией;
- BLS без генерации.

Метрики:
- throughput (RPS);
- latency (среднее, p50, p95, p99);
- Queue;
- Compute Input;
- Compute Infer;
- Compute Output;
- загрузка CPU и использование RAM.

Результаты будут опубликованы в reports/performance_report.md.

## Ограничения

Классификатор обучен на небольшом учебном наборе.
Его оценки качества не отражают production-производительность
на независимых реальных данных.

Малая LLM может генерировать неточные ответы.
Система предназначена для демонстрации инфраструктуры,
а не для реальной клиентской поддержки.

Все нагрузочные измерения относятся к CPU-конфигурации
и не должны напрямую сравниваться с GPU без учёта аппаратных различий.

## Статус

В разработке. Локальная ML-цепочка проверена.
Серверное развёртывание и нагрузочные тесты ещё не завершены.

## Загрузка внешних моделей

Для работы системы необходимы внешние веса моделей.

### Qwen2.5-0.5B-Instruct

Установите huggingface-hub:

    pip install huggingface-hub

Скачайте GGUF:

    python scripts/download_models.py

Файл будет помещён в:

    model_repository/llm_generator/1/qwen2.5-0.5b-instruct-q4_k_m.gguf

### Multilingual E5

Модель intfloat/multilingual-e5-small автоматически загружается
из Hugging Face при первом запуске embedding_model.

Для загрузки необходимо интернет-соединение.
После скачивания веса кешируются.

## Запуск через Docker Compose

После подготовки весов выполните:

    docker compose build
    docker compose up -d

Проверка готовности сервера:

    curl http://127.0.0.1:8000/v2/health/ready

Проверка состояния моделей:

    curl -X POST http://127.0.0.1:8000/v2/repository/index \
      -H 'Content-Type: application/json' \
      -d '{}'

Все четыре модели должны получить статус READY.

Примечание: образ tritonserver:latest должен быть предварительно
собран из исходников NVIDIA Triton с поддержкой CPU,
ONNX Runtime и Python Backend.

## Результаты эксперимента

На CPU-сервере Yandex Cloud получены следующие результаты:

| Сценарий | Throughput |
|---|---:|
| ONNX classifier, concurrency 4 | 13 546,6 RPS |
| E5 embedding, concurrency 1 | 64,88 RPS |
| BLS без генерации, concurrency 1 | 59,71 RPS |
| Qwen GGUF, concurrency 1 | 0,909 RPS |
| BLS с генерацией, concurrency 1 | 0,869 RPS |

Полные результаты и методика приведены
в reports/performance_report.md.

Основной вывод: генерация текста становится главным
ограничением производительности CPU-пайплайна.
Условный вызов LLM через BLS позволяет значительно
сократить затраты на обработку обращений,
для которых генерация ответа не требуется.
