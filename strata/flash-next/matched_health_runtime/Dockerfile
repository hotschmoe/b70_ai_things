# SOURCE-ONLY recipe. Parent must verify the local base tag equals the exact
# health image and supply verified official26.22 NEO/IGC/GMM/loader .deb files.
# No build or image replacement has been performed by this recipe's author.
ARG BASE_HEALTH_REF
FROM ${BASE_HEALTH_REF}
USER root
COPY driver-packages/ /opt/driver-control-packages/
# Fail rather than install dependencies from an unpinned network repository.
# The supplied bundle must contain every changed GPU-runtime dependency.
RUN cd /opt/driver-control-packages && sha256sum -c SHA256SUMS && \
    dpkg --install /opt/driver-control-packages/*.deb
# PyTorch/SYCL/oneCCL are preserved from the existing health image. A parent
# before/after census must prove all non-GPU package/library identities unchanged.
