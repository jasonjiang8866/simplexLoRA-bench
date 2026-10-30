FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 HF_HOME=/workspace/huggingface TOKENIZERS_PARALLELISM=false
WORKDIR /opt/bench
COPY . .
RUN python -m venv --system-site-packages /opt/bench-venv && \
    /opt/bench-venv/bin/pip install --no-cache-dir ".[gpu,test]" && \
    /opt/bench-venv/bin/python -m pip check && \
    /opt/bench-venv/bin/python -m pytest -q -m "not cuda" && \
    /opt/bench-venv/bin/python -m simplexbench.compat --output /opt/cpu-compatibility.json
ENV PATH=/opt/bench-venv/bin:$PATH
ENTRYPOINT ["/bin/bash", "/opt/bench/runpod/start.sh"]
