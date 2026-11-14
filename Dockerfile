FROM tritonserver:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    CMAKE_ARGS="-DGGML_CUDA=OFF" \
    FORCE_CMAKE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip \
      python3-venv \
      build-essential \
      cmake \
      git \
      libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/triton-venv

ENV VIRTUAL_ENV=/opt/triton-venv \
    PATH="/opt/triton-venv/bin:${PATH}" \
    PYTHONPATH="/opt/triton-venv/lib/python3.12/site-packages"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch CPU-only, без CUDA-зависимостей
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-triton.txt /tmp/requirements-triton.txt

RUN pip install --no-cache-dir \
    -r /tmp/requirements-triton.txt

# Проверка библиотек внутри образа
RUN python -c "import numpy, torch, sentence_transformers, llama_cpp; print('Python dependencies OK'); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

WORKDIR /models

CMD ["tritonserver", "--model-repository=/models"]
