set -e
cd shared && python3 -c "import familydata as f; f.make('/srv/data')" && cd ..
mkdir -p /srv/data/.thumbnail-cache/v2
# the hidden cache that filled the drive (regenerated automatically by the viewer app)
i=0; while dd if=/dev/zero of=/srv/data/.thumbnail-cache/v2/blob$i bs=1M count=4 2>/dev/null; do i=$((i+1)); done || true
chmod -R a+rX /srv/data
