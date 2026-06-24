import json, glob, struct, re
D="/models/Qwen3.6-27B-W4A8-rtn"
c=json.load(open(f"{D}/config.json"))
q=c.get("quantization_config",{})
cg=q.get("config_groups",{})
print("quant: format=", q.get("format"), "| config_groups:")
for k,v in cg.items():
    w=v.get("weights",{}); 
    print(f"  {k}: bits={w.get('num_bits')} group_size={w.get('group_size')} strategy={w.get('strategy')} sym={w.get('symmetric')} | targets={v.get('targets')}")
print("IGNORE:", q.get("ignore"))
# scan safetensors headers for 4304-dim tensors (deduped by layer pattern)
found={}
for f in sorted(glob.glob(f"{D}/*.safetensors")):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q", fh.read(8))[0]; hdr=json.loads(fh.read(n))
    for name,meta in hdr.items():
        if name=="__metadata__": continue
        shp=meta.get("shape",[])
        if 4304 in shp:
            key=re.sub(r"\.\d+\.", ".N.", name)
            found[key]=tuple(shp)
print("=== weight tensors with a 4304 dim (deduped) ===")
for k,s in sorted(found.items()): print(f"  {k}: {s}")
