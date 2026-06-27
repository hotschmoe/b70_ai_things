# sglang-xpu:woq = sglang-xpu:bmg + auto-round-lib (woqgemm XPU int4) + the woq_shim auto-import.
# Lets sglang serve AutoRound int4 (auto_round:auto_gptq) on XPU via the fast woqgemm kernel
# instead of the CUDA-gated gptq_gemm. See sglang/PERF.md.
FROM sglang-xpu:bmg
RUN pip install --no-input --no-cache-dir auto-round-lib && \
    python -c "import auto_round_kernel; print('auto_round_kernel OK')"
COPY woq_shim.py /opt/venv/lib/python3.12/site-packages/woq_shim.py
RUN printf 'import woq_shim\n' > /opt/venv/lib/python3.12/site-packages/woq_shim.pth
