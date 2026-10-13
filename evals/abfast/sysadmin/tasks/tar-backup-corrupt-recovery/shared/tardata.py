import random
N = 48
def content(i, ver):
    r = random.Random(1000 * i + ver)
    return "".join(r.choice("abcdefghijklmnopqrstuvwxyz \n") for _ in range(r.randrange(6000, 12000))).encode()
def path(i): return f"Documents/{['work','home','taxes','letters'][i % 4]}/file{i:02d}.txt"
# July backup holds files 0..35 (version 0); September holds all 48 files, all with new content (version 1)
def july(): return {path(i): content(i, 0) for i in range(36)}
def sept(): return {path(i): content(i, 1) for i in range(N)}
CUT = 0.62   # fraction of the compressed September archive kept
