import numpy as np
import triton_python_backend_utils as pb_utils

from sentence_transformers import SentenceTransformer


class TritonPythonModel:

    def initialize(self, args):
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-small",
            device="cpu"
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(
                request, "TEXT"
            )

            texts = [
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
                for value in text_tensor.as_numpy().reshape(-1)
            ]

            # Те же правила, что использовались при обучении.
            texts = ["query: " + text for text in texts]

            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

            output_tensor = pb_utils.Tensor(
                "EMBEDDING",
                embeddings
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
            )

        return responses

    def finalize(self):
        self.model = None
