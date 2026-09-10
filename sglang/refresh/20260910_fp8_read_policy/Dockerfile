FROM b70/sglang-fp8-policy-base:f82a10b2c3d04f10b230ba299dced23455366f307d96bdce54a7143264b41a99
LABEL b70.sglang.read_policy="xpu-fp16-query-fp8-cache-half-compute" \
      b70.sglang.source="2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed"
COPY overlay/decode_attention.py /opt/venv/lib/python3.12/site-packages/sglang/kernels/ops/attention/decode_attention.py
COPY overlay/extend_attention.py /opt/venv/lib/python3.12/site-packages/sglang/kernels/ops/attention/extend_attention.py
