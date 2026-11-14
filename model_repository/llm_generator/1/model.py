from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils
from llama_cpp import Llama


class TritonPythonModel:

    def initialize(self, args):
        model_path = (
            Path(__file__).resolve().parent
            / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"GGUF model not found: {model_path}"
            )

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(
                request, "PROMPT"
            )

            value = tensor.as_numpy().reshape(-1)[0]
            prompt = (
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
            )

            result = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты вежливый помощник службы "
                            "поддержки. Отвечай кратко по-русски. "
                            "Не придумывай сведения о заказах."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=96,
                temperature=0.2,
            )

            answer = result["choices"][0]["message"]["content"]
            answer = answer or ""

            output = pb_utils.Tensor(
                "RESPONSE",
                np.array([answer.encode("utf-8")], dtype=object)
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[output]
                )
            )

        return responses

    def finalize(self):
        self.llm = None
