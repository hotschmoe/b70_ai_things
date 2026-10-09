ARG BASE_IMAGE=b70/flashnext-llamacpp-runtime:neo26.22
FROM ${BASE_IMAGE}
USER root
COPY probe.py /opt/b70-c1-probe.py
COPY requirements.txt /opt/b70-c1-requirements.txt
RUN python3 /opt/b70-c1-probe.py --gpu-libraries > /opt/b70-c1-gpu-before.json
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/b70-c1-python && /opt/b70-c1-python/bin/python -m pip install --no-cache-dir --no-deps -r /opt/b70-c1-requirements.txt
RUN /opt/b70-c1-python/bin/python /opt/b70-c1-probe.py --receipt /opt/b70-c1-runtime.json --before /opt/b70-c1-gpu-before.json
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/bin/bash", "-lc"]
