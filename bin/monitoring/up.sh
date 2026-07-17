#!/usr/bin/env bash
# bin/monitoring/up.sh -- (re)create the b70 monitoring stack: Prometheus (scrapes the DD :18080/metrics)
# + Grafana (:3001, anonymous viewer, provisioned vllm/sglang dashboards). Idempotent: docker rm -f + run.
# No sudo, no root-in-container. Containers are --restart unless-stopped (survive reboot).
#
# WHY THIS EXISTS (2026-07-17): these were started by hand (no script), so when the docker/containerd
# bounce (the /var/lib/containerd -> 8TB move) disturbed their layers they crashed and recovery was manual.
# Prometheus gotcha: the old root-owned docker named volume was NOT writable by prometheus's `nobody` user
# ("open /prometheus/queries.active: permission denied") -> we bind-mount a host dir WE own and run as our
# uid instead. Grafana gotcha: its sqlite db lives in the writable layer (ephemeral) -> a fresh container
# re-provisions datasources+dashboards from the repo, so recreation is lossless for config.
set -uo pipefail
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
M="$REPO/bin/monitoring"
PROM_DATA="${PROM_DATA:-/mnt/vm_8tb/b70/prometheus_data}"   # host-owned (uid 1000); Prometheus TSDB
mkdir -p "$PROM_DATA"

echo "=== prometheus (host net, scrapes 127.0.0.1:18080, TSDB -> $PROM_DATA) ==="
docker rm -f b70_prometheus >/dev/null 2>&1 || true
docker run -d --name b70_prometheus --restart unless-stopped --network host --user "$(id -u):$(id -g)" \
  -v "$M/prometheus.yaml:/etc/prometheus/prometheus.yml" \
  -v "$PROM_DATA:/prometheus" \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus >/dev/null && echo "  prometheus up"

echo "=== grafana (:3001, anonymous viewer, provisioned) ==="
docker rm -f b70_grafana >/dev/null 2>&1 || true
docker run -d --name b70_grafana --restart unless-stopped --network host --user 472 \
  -e GF_SERVER_HTTP_PORT=3001 -e GF_AUTH_ANONYMOUS_ENABLED=true -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
  -e GF_AUTH_BASIC_ENABLED=false -e GF_USERS_ALLOW_SIGN_UP=false \
  -e GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/vllm-dashboard.json \
  -e GF_PATHS_PROVISIONING=/etc/grafana/provisioning \
  -v "$M/grafana/dashboards/json:/var/lib/grafana/dashboards" \
  -v "$M/grafana/dashboards/config:/etc/grafana/provisioning/dashboards" \
  -v "$M/grafana/datasources:/etc/grafana/provisioning/datasources" \
  grafana/grafana:latest >/dev/null && echo "  grafana up -> http://localhost:3001"

echo "=== verify (give it ~15s to scrape) ==="
echo "  targets:  curl -s http://localhost:9090/api/v1/targets | grep -o '\"health\":\"[a-z]*\"'"
echo "  hitrate:  curl -s 'http://localhost:9090/api/v1/query?query=100*sum(vllm:prefix_cache_hits_total)/sum(vllm:prefix_cache_queries_total)'"
