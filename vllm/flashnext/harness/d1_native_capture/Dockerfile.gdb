ARG D1_BASE_IMAGE
FROM ${D1_BASE_IMAGE}
USER root
# This is a diagnostic derivative only. Abort if apt changes any package
# already installed in the serving image; adding GDB must not upgrade the
# interpreter, libc, compiler/runtime or serving stack.
RUN dpkg-query -W -f '${binary:Package}\t${Version}\n' > /opt/d1-base-packages.tsv
COPY check_plan.py /opt/d1-check-plan.py
RUN apt-get update && apt-get --simulate --no-install-recommends --no-upgrade install gdb > /opt/d1-gdb-apt-plan.txt
RUN /usr/bin/python3 /opt/d1-check-plan.py
RUN apt-get install -y --no-install-recommends --no-upgrade gdb
COPY check_packages.py /opt/d1-check-packages.py
RUN /usr/bin/python3 /opt/d1-check-packages.py
