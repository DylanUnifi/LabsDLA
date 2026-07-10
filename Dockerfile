FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

WORKDIR /workspace

# Base dependencies including libgl1 for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip

# Install Python dependencies (single layer for better caching)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyterlab ipykernel && \
    python -m ipykernel install --user --name=dla_env --display-name "Python (DLA)"

ENV OMP_NUM_THREADS=32
ENV MKL_NUM_THREADS=32

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
