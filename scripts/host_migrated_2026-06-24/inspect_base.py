import json, glob, struct, re
D="/models/Qwen_Qwen3.6-27B"
# find tensors with a 4304 dim + collect all Linear-weight input dims not divisible by 128
found={}; bad_in={}
for f in sorted(glob.glob(f"{D}/*.safetensors")):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q", fh.read(8))[0]; hdr=json.loads(fh.read(n))
    for name,meta in hdr.items():
        if name=="__metadata__" or not name.endswith(".weight"): continue
        shp=meta.get("shape",[])
        if len(shp)==2:
            in_dim=shp[1]
            if in_dim % 128 != 0:
                key=re.sub(r"\.\d+\.", ".N.", name)
                bad_in.setdefault(key, (tuple(shp), in_dim))
        if 4304 in shp:
            key=re.sub(r"\.\d+\.", ".N.", name); found[key]=tuple(shp)
print("=== tensors with a 4304 dim ===")
for k,s in sorted(found.items()): print(f"  {k}: {s}")
print("=== Linear .weight with input dim NOT divisible by 128 (the group-128 blockers) ===")
for k,(s,ind) in sorted(bad_in.items()): print(f"  {k}: shape={s} in_dim={ind} (in/128={ind/128:.2f})")
