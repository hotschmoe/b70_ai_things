import random, os
# (relative path, seed, size) -- the family's real files
FILES = [(f"family/photos/img{i:02d}.jpg", 100 + i, 400_000 + i * 1000) for i in range(8)] + \
        [(f"family/docs/doc{i}.pdf", 200 + i, 120_000) for i in range(4)]
def make(root):
    for rel, seed, size in FILES:
        p = os.path.join(root, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(random.Random(seed).randbytes(size))
def ok(root):
    for rel, seed, size in FILES:
        p = os.path.join(root, rel)
        if not os.path.isfile(p) or open(p, "rb").read() != random.Random(seed).randbytes(size):
            return False
    return True
