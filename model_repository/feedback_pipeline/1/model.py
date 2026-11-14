import json

import numpy as np
import triton_python_backend_utils as pb_utils


CLASSES = ["complaint", "positive", "question"]


class TritonPythonModel:

    def initialize(self, args):
        print("Feedback BLS pipeline initialized")

    def call_model(self, model_name, inputs, output_name):
        """
        Внутренний вызов модели Triton через BLS.
        """
        request = pb_utils.InferenceRequest(
            model_name=model_name,
            requested_output_names=[output_name],
            inputs=inputs,
        )

        response = request.exec()

        if response.has_error():
            raise pb_utils.TritonModelException(
                response.error().message()
            )

        output = pb_utils.get_output_tensor_by_name(
            response, output_name
        )

        if output is None:
            raise pb_utils.TritonModelException(
                f"Missing output {output_name} from {model_name}"
            )

        return output.as_numpy()

    def process_text(self, text):
        # 1. Получаем эмбеддинг.
        text_tensor = pb_utils.Tensor(
            "TEXT",
            np.array([[text.encode("utf-8")]], dtype=object),
        )

        embedding = self.call_model(
            model_name="embedding_model",
            inputs=[text_tensor],
            output_name="EMBEDDING",
        )

        # Ожидаем [1, 384].
        if embedding.shape != (1, 384):
            raise ValueError(
                f"Unexpected embedding shape: {embedding.shape}"
            )

        # 2. Классифицируем эмбеддинг.
        embedding_tensor = pb_utils.Tensor(
            "embedding",
            embedding.astype(np.float32),
        )

        probabilities = self.call_model(
            model_name="classifier",
            inputs=[embedding_tensor],
            output_name="probabilities",
        )

        scores = probabilities.reshape(-1)

        if scores.size != len(CLASSES):
            raise ValueError(
                f"Unexpected classifier output: {probabilities.shape}"
            )

        class_index = int(np.argmax(scores))
        category = CLASSES[class_index]
        confidence = float(scores[class_index])

        # 3. Принимаем решение о генерации.
        requires_response = category in (
            "complaint",
            "question",
        )

        answer = None

        if requires_response:
            prompt_tensor = pb_utils.Tensor(
                "PROMPT",
                np.array([text.encode("utf-8")], dtype=object),
            )

            generated = self.call_model(
                model_name="llm_generator",
                inputs=[prompt_tensor],
                output_name="RESPONSE",
            )

            raw_answer = generated.reshape(-1)[0]

            answer = (
                raw_answer.decode("utf-8")
                if isinstance(raw_answer, bytes)
                else str(raw_answer)
            )

        # 4. Собираем единый ответ.
        return {
            "text": text,
            "category": category,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(score), 4)
                for name, score in zip(CLASSES, scores)
            },
            "requires_response": requires_response,
            "response": answer,
        }

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                tensor = pb_utils.get_input_tensor_by_name(
                    request, "TEXT"
                )

                if tensor is None:
                    raise ValueError("Missing TEXT input")

                values = tensor.as_numpy().reshape(-1)

                if len(values) != 1:
                    raise ValueError(
                        "Exactly one text is required"
                    )

                value = values[0]

                text = (
                    value.decode("utf-8")
                    if isinstance(value, bytes)
                    else str(value)
                )

                if not text.strip():
                    raise ValueError("Text must not be empty")

                result = self.process_text(text)

                output = pb_utils.Tensor(
                    "RESULT",
                    np.array(
                        [
                            json.dumps(
                                result,
                                ensure_ascii=False,
                            ).encode("utf-8")
                        ],
                        dtype=object,
                    ),
                )

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[output]
                    )
                )

            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(exc))
                    )
                )

        return responses

    def finalize(self):
        print("Feedback BLS pipeline finalized")
