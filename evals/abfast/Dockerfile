# abfast task image: Pi 1.2.0 + node + zig + python test deps. Build with:
#   docker build -t abfast:1 -f evals/abfast/Dockerfile \
#     --build-context node=$HOME/.nvm/versions/node/v24.17.0 \
#     --build-context zig=$HOME/.local/zig \
#     --build-context pi=/mnt/vm_8tb/b70/tooling/pi-1.2.0 evals/abfast
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git patch python3 python3-venv python3-pip build-essential ca-certificates \
      curl ripgrep jq less tree procps xz-utils && rm -rf /var/lib/apt/lists/*
COPY --from=node / /opt/node
COPY --from=zig / /opt/zig
COPY --from=pi / /opt/pi
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir \
      pytest pytest-timeout pytest-mock hypothesis freezegun pretend tomli-w watchdog \
      ephemeral-port-reserve cloudpickle pympler zope.interface typing-extensions \
      setuptools wheel trustme cryptography greenlet packaging markupsafe jinja2 itsdangerous colorama
ENV PATH=/opt/venv/bin:/opt/node/bin:/opt/zig:/opt/pi/node_modules/.bin:$PATH \
    PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 CI=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/agent PI_CODING_AGENT_DIR=/home/agent/.pi-agent
RUN useradd -m -u 1000 -d /home/agent agent 2>/dev/null || usermod -l agent -d /home/agent -m ubuntu; \
    mkdir -p /work /logs && chown -R 1000:1000 /work /logs /home/agent
USER 1000
WORKDIR /work
