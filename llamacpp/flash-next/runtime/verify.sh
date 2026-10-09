#!/usr/bin/env bash
# CPU-only package/linkage audit. Never initializes a GPU or enumerates devices.
set -euo pipefail
export LC_ALL=C
record=/opt/flashnext-runtime
dpkg-query -W > "$record/packages-installed.txt"
for package in libze-intel-gpu1 intel-opencl-icd intel-ocloc; do
    [[ $(dpkg-query -W -f='${Version}' "$package") == 26.22.38646.4-0 ]]
done
for package in intel-igc-core-2 intel-igc-opencl-2; do
    [[ $(dpkg-query -W -f='${Version}' "$package") == 2.36.3 ]]
done
[[ $(dpkg-query -W -f='${Version}' libigdgmm12) == 22.10.0 ]]
# Keep the base image's Level Zero loader: upstream 1.28.2 matches the host,
# while its Ubuntu24.04 package revision differs from the host's Ubuntu26.04.
[[ $(dpkg-query -W -f='${Version}' libze1) == 1.28.2-1~24.04~ppa1 ]]
for package in libigc2 libigdfcl2; do
    [[ $(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true) != installed ]]
done
ldconfig -p > "$record/ldconfig.txt"
for library in libigc.so.2 libigdfcl.so.2 libiga64.so.2; do
    grep -F "$library " "$record/ldconfig.txt" | grep -F '/usr/local/lib/'
done
ldd /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1 > "$record/level-zero-linkage.txt"
if grep -q 'not found' "$record/level-zero-linkage.txt"; then
    cat "$record/level-zero-linkage.txt" >&2
    exit 2
fi
sha256sum /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1 \
    /usr/lib/x86_64-linux-gnu/libigdgmm.so.12 \
    /usr/local/lib/libigc.so.2 /usr/local/lib/libigdfcl.so.2 \
    > "$record/runtime-sha256.txt"
echo 'CONFIG NEO26.22.38646.4 IGC2.36.3 GMM22.10.0 loader1.28.2'
echo 'RESULT package and linker audit passed; no GPU execution'
echo 'VERDICT runtime image prepared; qualification still required'
