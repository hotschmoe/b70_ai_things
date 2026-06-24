import json, glob, struct, re
D="/models/Qwen3.6-27B-W4A8-rtn"
DT={"BF16":2,"F16":2,"I8":1,"F32":4,"I32":4,"I64":8,"U8":1}
cat={}  # category -> [bytes, dtypes set]
for f in sorted(glob.glob(f"{D}/*.safetensors")):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n))
    for name,meta in h.items():
        if name=="__metadata__": continue
        shp=meta.get("shape",[]); dt=meta["dtype"]
        nb=1
        for s in shp: nb*=s
        nb*=DT.get(dt,2)
        if "visual" in name: c="visual"
        elif "linear_attn" in name: c="GDN_linear_attn"
        elif "lm_head" in name: c="lm_head"
        elif "mtp" in name: c="mtp"
        elif "scale" in name: c="scales"
        elif re.search(r"language_model.*\.(mlp|self_attn)\.", name): c="text_mlp_attn(quant)"
        else: c="other"
        e=cat.setdefault(c,[0,set()]); e[0]+=nb; e[1].add(dt)
print(f"{'category':24} {'GiB':>7}  dtypes")
tot=0
for c,(b,dts) in sorted(cat.items(), key=lambda x:-x[1][0]):
    tot+=b; print(f"{c:24} {b/2**30:7.2f}  {sorted(dts)}")
print(f"{'TOTAL':24} {tot/2**30:7.2f}")
