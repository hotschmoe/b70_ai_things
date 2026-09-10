FROM sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1
ARG NATIVE_SHA
COPY _xpu_C.abi3.so /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so
LABEL neural.download.xpu.extension.sha256="${NATIVE_SHA}" \
      b70.native.conv.contract="upstream5802-on-r276-r35-r50-r221" \
      b70.native.conv.upstream="5802a414d47855b01b63121bce3655795ef8dfa8" \
      b70.native.parent.sha256="271db0d4882124e21ac6a4d080bfeab303fbb08b9ec10e11f21d10fb0723998f"
