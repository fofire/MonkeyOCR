# Multi-stage build for smaller final image with models baked in
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-dev python3-pip build-essential git curl wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic links
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Configure pip for faster installs
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
ARG CUDA_VERSION=126
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

RUN pip install --no-cache-dir \
    paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu${CUDA_VERSION}/

RUN pip install --no-cache-dir \
    lmdeploy==0.9.2 huggingface_hub paddlex[base] vllm==0.9.1

COPY . /tmp/MonkeyOCR
WORKDIR /tmp/MonkeyOCR

# Create model directory and set environment variables for download
ENV HF_HUB_CACHE=/tmp/MonkeyOCR/model_weight
ENV MODELSCOPE_CACHE=/tmp/MonkeyOCR/model_weight
RUN mkdir -p /tmp/MonkeyOCR/model_weight

# Download models during build (this is the key change)
RUN chmod +x docker/download_models.sh && \
    ./docker/download_models.sh || \
    (echo "Model download failed, trying alternative method..." && \
     python tools/download_model.py -t modelscope || \
     python tools/download_model.py)

RUN pip install --no-cache-dir -e .

# Apply LMDeploy patcher if needed
ARG LMDEPLOY_PATCHED=true
RUN if [ "$LMDEPLOY_PATCHED" = "true" ]; then \
      python tools/lmdeploy_patcher.py patch && \
      echo "Successfully applied lmdeploy patch"; \
    else \
      echo "LMDEPLOY_PATCHED is false, skipping patch"; \
    fi

# Final runtime stage
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-dev python3-pip ccache build-essential nano \
    libgl1-mesa-glx libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create symbolic links
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN useradd -m -s /bin/bash appuser \
    && mkdir -p /app/MonkeyOCR/model_weight \
    && chown -R appuser:appuser /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . /app/MonkeyOCR

# Copy pre-downloaded models from builder stage (key addition)
COPY --from=builder --chown=appuser:appuser /tmp/MonkeyOCR/model_weight /app/MonkeyOCR/model_weight

WORKDIR /app/MonkeyOCR

# Copy and set permissions for scripts
COPY --chown=appuser:appuser docker/entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER appuser

# Default command
CMD ["./entrypoint.sh"]