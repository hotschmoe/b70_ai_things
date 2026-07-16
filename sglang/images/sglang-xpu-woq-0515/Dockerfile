# sglang-xpu:woq-0515 = sglang-xpu:bmg-0515 (v0.5.15.post1 base) + auto-round-lib (woqgemm XPU int4)
# + the woq_shim auto-import. 0.5.15 re-graft of sglang-xpu:woq. See sglang/SGLANG_0515_UPGRADE.md.
#
# Shims copied here are the CANONICAL sglang/patches/ versions (monkeypatch installers -- validated
# against v0.5.15.post1 source; all patch targets still resolve). fused_recurrent.py / fused_gdn_gating.py
# are byte-identical to upstream 0.5.15 (our copies add only an inert B70_GDN_DECODE_WARPS env knob;
# PERF.md: NO warm speedup) so baking them is a zero-drift no-op.
FROM sglang-xpu:bmg-0515
RUN pip install --no-input --no-cache-dir auto-round-lib && \
    python -c "import auto_round_kernel; print('auto_round_kernel OK')"
COPY woq_shim.py /opt/venv/lib/python3.12/site-packages/woq_shim.py
COPY mtp_tree_xpu.py /opt/venv/lib/python3.12/site-packages/mtp_tree_xpu.py
COPY w8a8_shim.py /opt/venv/lib/python3.12/site-packages/w8a8_shim.py
# W4A8/W4A16 hybrid shim (env-gated B70_XPU_W4A8=1; needs the built _xpu_C.abi3.so via B70_XPU_C_SO).
COPY w4a8_shim.py /opt/venv/lib/python3.12/site-packages/w4a8_shim.py
RUN printf 'import woq_shim\n' > /opt/venv/lib/python3.12/site-packages/woq_shim.pth
# GDN decode kernels: num_warps env-tunable (default 1 = upstream). NO warm speedup. Byte-identical to
# upstream 0.5.15 aside from the inert knob.
COPY fused_recurrent.py /opt/venv/lib/python3.12/site-packages/sglang/srt/layers/attention/fla/fused_recurrent.py
COPY fused_gdn_gating.py /opt/venv/lib/python3.12/site-packages/sglang/srt/layers/attention/fla/fused_gdn_gating.py
