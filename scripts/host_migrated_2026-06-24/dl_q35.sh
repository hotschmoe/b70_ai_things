docker run --rm -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70 -e HF_HOME=/mnt/vm_8tb/b70/hf_cache \
  --entrypoint bash vllm-xpu-env:int8 -lc 'python3 /mnt/vm_8tb/b70/dl_q35.py'
