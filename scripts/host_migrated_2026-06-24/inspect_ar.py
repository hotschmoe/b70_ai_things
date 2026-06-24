import auto_round, os
root = os.path.dirname(auto_round.__file__)
needle = "only support to export llm_compressor"
for dirpath, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        try:
            src = open(p).read().splitlines()
        except Exception:
            continue
        for i, l in enumerate(src):
            if needle in l:
                print("FILE", p)
                for j in range(max(0, i - 22), min(len(src), i + 3)):
                    print(j + 1, src[j])
                print("=====")
