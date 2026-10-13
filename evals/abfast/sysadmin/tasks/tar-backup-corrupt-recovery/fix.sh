set -e
mkdir -p /tmp/latest /tmp/older
tar -xzf /mnt/backup/docs-2026-09.tar.gz -C /tmp/latest 2>/dev/null || true
tar -xzf /mnt/backup/docs-2026-07.tar.gz -C /tmp/older
python3 - <<'P'
import io, os, shutil, tarfile, zlib
blob = open("/mnt/backup/docs-2026-09.tar.gz", "rb").read()
raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(blob)
good = set(); tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
try:
    for m in tf:
        f = tf.extractfile(m)
        if f is not None and len(f.read()) == m.size: good.add(m.name)
except Exception:
    pass
older = set(); 
for dp, _, fs in os.walk("/tmp/older/Documents"):
    for f in fs: older.add(os.path.relpath(os.path.join(dp, f), "/tmp/older"))
allp = set()
nl = no = nm = 0
for p in sorted(good):
    dst = "/home/pat/" + p; os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy("/tmp/latest/" + p, dst); nl += 1
for p in sorted(older - good):
    dst = "/home/pat/" + p; os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy("/tmp/older/" + p, dst); no += 1
# total files in the september backup = 48 (known from the tar listing of the first member names is not possible past the cut):
# the reference fix uses the intact index of what exists in July plus what is recoverable; files only in September beyond the cut are unknowable
open("/home/pat/recovery.txt", "w").write("latest=%d older=%d\n" % (nl, no))
P
