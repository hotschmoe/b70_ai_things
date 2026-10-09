# Supply the already-recorded compiler image; no compiler/toolkit upgrade here.
ARG BUILD_IMAGE
FROM ${BUILD_IMAGE}
USER root
COPY packages.json /opt/flashnext-runtime/packages.json
COPY install.py /opt/flashnext-runtime/install.py
COPY verify.sh /opt/flashnext-runtime/verify.sh
RUN python3 /opt/flashnext-runtime/install.py \
    && bash /opt/flashnext-runtime/verify.sh
