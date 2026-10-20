# Runtime overlay imported because xpu-run.sh puts this directory on
# PYTHONPATH (site.py imports sitecustomize from sys.path).
# Add small XPU workarounds here. Keep this file import-safe: a failure
# here breaks every Python process in the container.
