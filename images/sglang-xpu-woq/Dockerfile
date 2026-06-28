# sglang-xpu:woq = sglang-xpu:bmg + auto-round-lib (woqgemm XPU int4) + the woq_shim auto-import
# + env-tunable GDN decode num_warps (B70_GDN_DECODE_WARPS; NO-OP warm -- see PERF.md correction). See sglang/PERF.md.
FROM sglang-xpu:bmg
RUN pip install --no-input --no-cache-dir auto-round-lib && \
    python -c "import auto_round_kernel; print('auto_round_kernel OK')"
COPY woq_shim.py /opt/venv/lib/python3.12/site-packages/woq_shim.py
COPY mtp_tree_xpu.py /opt/venv/lib/python3.12/site-packages/mtp_tree_xpu.py
COPY w8a8_shim.py /opt/venv/lib/python3.12/site-packages/w8a8_shim.py
# W4A8/W4A16 hybrid shim (env-gated B70_XPU_W4A8=1; needs the built _xpu_C.abi3.so via B70_XPU_C_SO).
COPY w4a8_shim.py /opt/venv/lib/python3.12/site-packages/w4a8_shim.py
RUN printf 'import woq_shim\n' > /opt/venv/lib/python3.12/site-packages/woq_shim.pth
# GDN decode kernels: num_warps env-tunable (default 1 = upstream). NO warm speedup (the "1.69x" was a cold-bench artifact).
COPY fused_recurrent.py /opt/venv/lib/python3.12/site-packages/sglang/srt/layers/attention/fla/fused_recurrent.py
COPY fused_gdn_gating.py /opt/venv/lib/python3.12/site-packages/sglang/srt/layers/attention/fla/fused_gdn_gating.py
