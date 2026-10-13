set -e
mkdir -p /mnt/backup /home/pat/Documents
cd shared && python3 - <<'P'
import io, tarfile, gzip, tardata
def mk(files, out):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for p, d in files.items():
            ti = tarfile.TarInfo(p); ti.size = len(d); ti.mtime = 1790000000; ti.uid = ti.gid = 1000; ti.uname = ti.gname = "pat"; ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(d))
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0, compresslevel=6) as g: g.write(raw.getvalue())
    return gz.getvalue()
open("/mnt/backup/docs-2026-07.tar.gz", "wb").write(mk(tardata.july(), None))
s = mk(tardata.sept(), None)
open("/mnt/backup/docs-2026-09.tar.gz", "wb").write(s[:int(len(s) * tardata.CUT)])
P
chown -R pat:pat /home/pat
chmod -R a-w /mnt/backup 2>/dev/null || true
