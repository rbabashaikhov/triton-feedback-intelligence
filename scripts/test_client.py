import argparse
import json

import numpy as np
import tritonclient.http as httpclient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton HTTP endpoint",
    )
    parser.add_argument(
        "--text",
        default="Когда доставят мой заказ?",
    )
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(
        url=args.url,
        verbose=False,
    )

    print("Server ready:", client.is_server_ready())

    if not client.is_model_ready("feedback_pipeline"):
        raise RuntimeError("feedback_pipeline is not READY")

    text_data = np.array(
        [args.text.encode("utf-8")],
        dtype=object,
    )

    input_tensor = httpclient.InferInput(
        "TEXT",
        [1],
        "BYTES",
    )
    input_tensor.set_data_from_numpy(text_data)

    result = client.infer(
        model_name="feedback_pipeline",
        inputs=[input_tensor],
        outputs=[
            httpclient.InferRequestedOutput("RESULT")
        ],
    )

    raw = result.as_numpy("RESULT")[0]
    response = json.loads(raw.decode("utf-8"))

    print(json.dumps(
        response,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
