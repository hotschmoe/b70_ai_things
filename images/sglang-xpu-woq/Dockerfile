# sglang-xpu:woq = sglang-xpu:bmg + auto-round-lib (woqgemm XPU int4) + the woq_shim auto-import
# + the GDN decode-kernel occupancy fix (num_warps 1->4 = 1.69x single-card decode). See sglang/PERF.md.
FROM sglang-xpu:bmg
RUN pip install --no-input --no-cache-dir auto-round-lib && \
    python -c "import auto_round_kernel; print('auto_round_kernel OK')"
COPY woq_shim.py /opt/venv/lib/python3.12/site-packages/woq_shim.py
RUN printf 'import woq_shim\n' > /opt/venv/lib/python3.12/site-packages/woq_shim.pth
# GDN decode kernels: num_warps tunable via B70_GDN_DECODE_WARPS (default 4; was hardcoded 1 -> occupancy-starved)
COPY fused_recurrent.py /opt/venv/lib/python3.12/site-packages/sglang/srt/layers/attention/fla/fused_recurrent.py
COPY fused_gdn_gating.py /opt/venv/lib/python3.12/site-packages/sglang/srt/layers/attention/fla/fused_gdn_gating.py
