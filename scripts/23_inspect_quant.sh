#!/usr/bin/env bash
# What quantization methods + INT8 kernels does llm-scaler b8.3.1 actually support
# on XPU? Decides which W8A8 format engages the B70 XMX INT8 fast path.
IMG="intel/llm-scaler-vllm:0.14.0-b8.3.1"

echo "===== --quantization choices ====="
docker run --rm --entrypoint bash "$IMG" -c 'vllm serve --help 2>/dev/null | grep -iE -A40 "^\s*--quantization" | head -45'

echo; echo "===== registered quant methods ====="
docker run --rm --entrypoint python "$IMG" -c '
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
print(sorted(QUANTIZATION_METHODS))
' 2>&1 | tail -5

echo; echo "===== intel int8/int4 multi-arc kernels present ====="
docker run --rm --entrypoint bash "$IMG" -c 'ls /usr/local/lib/python3.12/dist-packages/ | grep -iE "int8|int4|quark|multi_arc|woq|gemm"'

echo; echo "===== ipex_quant supported schemes (grep) ====="
docker run --rm --entrypoint bash "$IMG" -c 'grep -iE "sym_int8|w8a8|int8|per_channel|per_token|fp8|sym_int4|quark|compressed" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/ipex_quant.py 2>/dev/null | head -25'

echo; echo "===== DONE ====="
