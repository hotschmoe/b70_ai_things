FROM intel/sglang-dev@sha256:1a13d3d2b0adc63422a643eb5c59524842577d99ca3935b4b8c2c432e739e127
LABEL b70.sglang.source="2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed" \
      b70.sglang.kernel_source="3802d7d87829073535fe5f84d3aef4ed76e7ef90" \
      b70.sglang.profile="triton-dense-bmg-no-fmha-no-moe-no-mla-gdn-on"
COPY wheels/ /tmp/b70-main-wheels/
RUN python3 -m pip uninstall -y sgl-kernel && \
    python3 -m pip install --no-deps --force-reinstall /tmp/b70-main-wheels/*.whl && \
    rm -rf /tmp/b70-main-wheels
