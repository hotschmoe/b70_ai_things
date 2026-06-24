from huggingface_hub import snapshot_download
p = snapshot_download("nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8",
                      local_dir="/mnt/vm_8tb/b70/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8")
print("DL35_DONE", p)
