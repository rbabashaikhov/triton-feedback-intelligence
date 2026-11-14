import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "inputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-small"

COMPLAINTS = [
    "Доставка задерживается уже неделю.",
    "Товар пришёл повреждённым.",
    "Мне отправили неправильный заказ.",
    "Я не получил деньги за возврат.",
    "Служба поддержки не отвечает.",
]

QUESTIONS = [
    "Когда доставят мой заказ?",
    "Как оформить возврат товара?",
    "Можно ли изменить адрес доставки?",
    "Какие способы оплаты доступны?",
    "Как проверить статус заказа?",
]

POSITIVES = [
    "Отличный магазин, спасибо!",
    "Доставка прошла очень быстро.",
    "Товар полностью соответствует описанию.",
    "Очень доволен покупкой.",
    "Прекрасное обслуживание, рекомендую.",
]


def save(name, data):
    path = OUTPUT / name

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {"data": data},
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {path.name} ({len(data)} requests)")


def text_requests(texts, input_name):
    return [
        {input_name: [text]}
        for text in texts
    ]


def main():
    all_texts = COMPLAINTS + QUESTIONS + POSITIVES
    needs_response = COMPLAINTS + QUESTIONS

    # Отдельная модель эмбеддингов.
    save(
        "embedding.json",
        text_requests(all_texts, "TEXT"),
    )

    # Генеративная модель.
    save(
        "generation.json",
        text_requests(needs_response, "PROMPT"),
    )

    # Полный BLS-пайплайн.
    save(
        "pipeline_all.json",
        text_requests(all_texts, "TEXT"),
    )

    # BLS с генерацией.
    save(
        "pipeline_generation.json",
        text_requests(needs_response, "TEXT"),
    )

    # BLS без предполагаемой генерации.
    save(
        "pipeline_positive.json",
        text_requests(POSITIVES, "TEXT"),
    )

    # Для классификатора нужны настоящие эмбеддинги.
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    vectors = model.encode(
        ["query: " + text for text in all_texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    assert vectors.shape == (len(all_texts), 384)

    save(
        "classifier.json",
        [
            {"embedding": vector.tolist()}
            for vector in vectors
        ],
    )

    print("\nBENCHMARK DATA READY")


if __name__ == "__main__":
    main()
