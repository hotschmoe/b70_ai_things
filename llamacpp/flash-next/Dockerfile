FROM intel/oneapi:2026.1.0-devel-ubuntu24.04@sha256:e9db518398753434ee5aab9740a25f1d3134396a30be1569cfad8f8b0d90740c

# No driver packages or host runtime changes. Record resolved package versions
# with the build; the compiler/toolkit base is pinned by digest.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates cmake ninja-build git python3 pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C
ENTRYPOINT ["/bin/bash", "-lc"]
