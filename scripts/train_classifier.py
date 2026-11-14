from pathlib import Path
import json

import numpy as np
import onnx
import onnxruntime as ort

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model_repository" / "classifier" / "1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-small"

# Учебный набор данных — демонстрация инфраструктуры,
# а не production-классификатор.

DATA = [
    ("Товар пришёл сломанным", "complaint"),
    ("Доставка задерживается уже неделю", "complaint"),
    ("Мне привезли неправильный заказ", "complaint"),
    ("Качество товара ужасное", "complaint"),
    ("Служба поддержки не отвечает", "complaint"),
    ("Деньги за возврат так и не пришли", "complaint"),
    ("Курьер опоздал на два часа", "complaint"),
    ("Упаковка была повреждена", "complaint"),
    ("Заказ отменили без объяснения", "complaint"),
    ("Приложение постоянно зависает", "complaint"),

    ("Когда доставят заказ?", "question"),
    ("Как оформить возврат?", "question"),
    ("Сколько стоит доставка?", "question"),
    ("Можно ли изменить адрес?", "question"),
    ("Где посмотреть статус заказа?", "question"),
    ("Какие способы оплаты доступны?", "question"),
    ("Как связаться с поддержкой?", "question"),
    ("Есть ли гарантия на товар?", "question"),
    ("Когда появится товар в продаже?", "question"),
    ("Можно ли оплатить при получении?", "question"),

    ("Отличный магазин, рекомендую", "positive"),
    ("Доставка была очень быстрой", "positive"),
    ("Товар полностью соответствует описанию", "positive"),
    ("Очень доволен покупкой", "positive"),
    ("Спасибо за прекрасное обслуживание", "positive"),
    ("Всё работает идеально", "positive"),
    ("Курьер был очень вежливым", "positive"),
    ("Качество товара отличное", "positive"),
    ("Обязательно закажу ещё", "positive"),
    ("Приятно удивлён скоростью доставки", "positive"),
]


def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    texts = ["query: " + text for text, _ in DATA]
    labels = [label for _, label in DATA]

    print("Generating embeddings...")
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    print("Embedding shape:", embeddings.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    print("Training classifier...")
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # Экспорт только вероятностей в ONNX.
    # zipmap=False делает выход обычным тензором.
    options = {id(classifier): {"zipmap": False}}

    onnx_model = convert_sklearn(
        classifier,
        initial_types=[
            ("embedding", FloatTensorType([None, 384]))
        ],
        options=options,
        target_opset=15,
    )

    # sklearn экспортирует label и probabilities.
    # Нам нужны только вероятности трёх классов.
    old_name = onnx_model.graph.output[1].name

    for node in onnx_model.graph.node:
        for i, name in enumerate(node.output):
            if name == old_name:
                node.output[i] = "probabilities"

    for value in onnx_model.graph.value_info:
        if value.name == old_name:
            value.name = "probabilities"

    onnx_model.graph.output[1].name = "probabilities"

    # Убираем выход label из интерфейса модели.
    del onnx_model.graph.output[0]

    output_path = MODEL_DIR / "model.onnx"
    onnx.save(onnx_model, output_path)

    onnx.checker.check_model(str(output_path))

    print("\nONNX model saved:", output_path)

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    print("\nONNX inputs:")
    for item in session.get_inputs():
        print(item.name, item.shape, item.type)

    print("\nONNX outputs:")
    for item in session.get_outputs():
        print(item.name, item.shape, item.type)

    sample = embeddings[:1]
    result = session.run(None, {"embedding": sample})

    print("\nInference test:")
    for output, value in zip(session.get_outputs(), result):
        print(output.name, np.asarray(value).shape)

    metadata = {
        "embedding_model": MODEL_NAME,
        "classes": classifier.classes_.tolist(),
        "embedding_dim": 384,
        "training_samples": len(DATA),
    }

    metadata_path = ROOT / "model_repository" / "classifier" / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nMetadata saved:", metadata_path)


if __name__ == "__main__":
    main()
