import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

TEXTS = [
    "Отличный магазин, спасибо!",
    "Когда доставят мой заказ?",
    "Товар пришёл повреждённым.",
]

metadata = json.loads(
    (ROOT / "model_repository/classifier/metadata.json").read_text()
)

print("1. Loading embeddings...")
embedder = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu",
)

print("2. Loading classifier...")
classifier = ort.InferenceSession(
    str(ROOT / "model_repository/classifier/1/model.onnx"),
    providers=["CPUExecutionProvider"],
)

print("3. Loading Qwen...")
llm = Llama(
    model_path=str(
        ROOT
        / "model_repository/llm_generator/1"
        / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ),
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False,
)

for text in TEXTS:
    print(f"\nINPUT: {text}")

    embedding = embedder.encode(
        ["query: " + text],
        normalize_embeddings=True,
    ).astype(np.float32)

    probabilities = classifier.run(
        ["probabilities"],
        {"embedding": embedding},
    )[0][0]

    idx = int(np.argmax(probabilities))
    category = metadata["classes"][idx]

    print("CATEGORY:", category)
    print("CONFIDENCE:", round(float(probabilities[idx]), 4))

    if category in ("complaint", "question"):
        result = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты вежливый помощник службы поддержки. "
                        "Отвечай кратко по-русски. "
                        "Не придумывай сведения о заказах."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=96,
            temperature=0.2,
        )

        answer = result["choices"][0]["message"]["content"]
        print("LLM RESPONSE:", answer)
    else:
        print("LLM SKIPPED")

print("\nLOCAL PIPELINE TEST COMPLETED")
