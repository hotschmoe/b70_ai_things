#!/usr/bin/env python3
"""Light soak monitor for the hotschmoe-dd endpoint (TensorFold or Strata backend).

Every SAMPLE_S seconds it reads new server/frontdoor log lines, polls /health on 127.0.0.1:18124 and reads
card telemetry from sysfs (no GPU lease, no device access). Every SUMMARY_S seconds it writes a summary
markdown file and appends to soak.jsonl. It records per-request metrics only: TTFT, decode tok/s, prompt,
cached and completion tokens, concurrency, MTP acceptance, finish reason and HTTP status. It never reads or
stores prompt or response text, request ids or token hashes.

Alerts (one line each) go to ALERTS.log: HTTP errors, device-loss/error lines, restarts, health failures,
and slowdowns (window p50 decode tok/s < 70% of the first-hour baseline, or p50 TTFT > 3x baseline).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

LOGS = Path(os.environ.get("SOAK_LOGS", "/mnt/vm_8tb/b70/logs/hotschmoe-dd"))
OUT = Path(os.environ.get("SOAK_OUT", "/mnt/vm_8tb/b70/results/hotschmoe-dd/soak"))
FROM_START = os.environ.get("SOAK_FROM_START") == "1"  # tests only
BACKEND_URL = "http://127.0.0.1:18124"
UNITS = {"tf": "hotschmoe-dd-tensorfold.service", "strata": "hotschmoe-dd.service"}
SERVER_LOG = {"tf": LOGS / "tf-server.log", "strata": LOGS / "server.log"}
FRONTDOOR_LOG = LOGS / "frontdoor.log"

TF_DONE = re.compile(
    r"\[tensorfold\] done \S+ prompt=(?P<prompt>\d+) cached=(?P<cached>\d+) thinking=(?P<thinking>\w+) "
    r"effort=(?P<effort>\S+) tokens=(?P<tokens>\d+) sha=\S+ finish=(?P<finish>\S+)"
    r"(?: rounds=(?P<rounds>\d+) accepted=(?P<acc>\d+)/(?P<drafted>\d+))?.*? ttft=(?P<ttft>[\d.]+) "
    r"decode=(?P<decode>[\d.]+) elapsed=(?P<elapsed>[\d.]+) queued=(?P<queued>[\d.]+) tok/s=(?P<tps>[\d.]+)")
TF_LANES = re.compile(r"flashnext: active lanes (\d+)")
HTTP_LINE = re.compile(r'"(?P<method>[A-Z]+) (?P<path>\S+) HTTP/[\d.]+" (?P<status>\d{3})')
STRATA_DONE = re.compile(r"\[strata\] done: (?P<tokens>\d+) tokens in (?P<secs>[\d.]+) s \((?P<tps>[\d.]+) tok/s\) \((?P<finish>[^,]+), cancel=(?P<cancel>\w+)\)")
STRATA_PROMPT = re.compile(r"\[strata\] reading the prompt: [\d,]+ of (?P<total>[\d,]+) tokens")
RESTART = re.compile(r"\[dd_launch_tf\] variant=|gpu-run: \[hotschmoe-dd[^\]]*\] acquired")
BAD = re.compile(r"DEVICE_LOST|ZE_RESULT_ERROR|ZeFailed|ze[A-Z]\w+ failed|WEDGED|HUNG|panic|Segmentation fault|"
                 r"core dumped|out of memory|OUT_OF_DEVICE_MEMORY|error\(|^error:|Traceback|"
                 r"FN_RECORD fallback|PoolExhausted|NoHeldDrafts|CapturedScratchResize", re.I)
IGNORE_BAD = re.compile(r"spills \d+ bytes of registers|libtfgrammar")


def now() -> float:
    return time.time()


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


class Tail:
    """Incremental reader that survives truncation/rotation; starts at EOF the first time."""

    def __init__(self, path: Path, state: dict) -> None:
        self.path, self.state, self.key = path, state, str(path)

    def read(self) -> list[str]:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return []
        pos = self.state.get(self.key)
        if pos is None or pos > st.st_size:
            pos = (0 if FROM_START else st.st_size) if pos is None else 0
        if st.st_size - pos > 64 << 20:  # never parse a huge backlog
            pos = st.st_size - (64 << 20)
        with self.path.open("rb") as f:
            f.seek(pos)
            data = f.read()
        end = data.rfind(b"\n") + 1
        self.state[self.key] = pos + end
        return data[:end].decode("utf-8", "replace").splitlines()


def cards() -> list[dict]:
    out = []
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]")):
        hw = [h for h in glob.glob(dev + "/device/hwmon/hwmon*") if Path(h, "name").read_text().strip() == "xe"]
        if not hw:
            continue
        h, rec = hw[0], {"card": os.path.basename(dev)}
        for i in range(1, 32):
            lab = Path(h, f"temp{i}_label")
            if lab.exists() and lab.read_text().strip() in ("pkg", "vram"):
                rec["temp_" + lab.read_text().strip() + "_c"] = int(Path(h, f"temp{i}_input").read_text()) / 1000
        try:
            rec["energy_uj"] = int(Path(h, "energy1_input").read_text())
        except OSError:
            pass
        for k, f in (("act_mhz", "act_freq"), ("throttle", "throttle/status")):
            try:
                rec[k] = int(Path(dev, "device/tile0/gt0/freq0", f).read_text())
            except (OSError, ValueError):
                pass
        out.append(rec)
    return out


def http_get(path: str, timeout: float = 3.0) -> tuple[int | None, dict | None]:
    try:
        with urllib.request.urlopen(BACKEND_URL + path, timeout=timeout) as r:
            body = r.read(65536)
            try:
                return r.status, json.loads(body)
            except ValueError:
                return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:  # noqa: BLE001
        return None, None


def unit_state(unit: str) -> dict:
    try:
        out = subprocess.run(["systemctl", "--user", "show", "-p", "ActiveState,NRestarts,ExecMainStartTimestampMonotonic", unit],
                             capture_output=True, text=True, timeout=5).stdout
        return dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    except Exception:  # noqa: BLE001
        return {}


class Monitor:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        OUT.mkdir(parents=True, exist_ok=True)
        self.state_path = OUT / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("offsets", {})
        self.backend = a.backend
        self.reset_window()
        self.lanes = 0
        self.pending_prompt: int | None = None
        self.health_fail = 0
        self.last_energy: dict[str, tuple[float, int]] = {}

    def reset_window(self) -> None:
        self.w = {"start": now(), "requests": [], "http": {}, "fd_http": {}, "errors": [], "restarts": 0,
                  "health_ok": 0, "health_fail": 0, "max_lanes": 0, "cards": {}, "clamp": None, "seen": {}}

    def alert(self, msg: str) -> None:
        seen = self.w.setdefault("seen", {})
        seen[msg] = seen.get(msg, 0) + 1
        if seen[msg] > 1:  # one ALERTS.log line per distinct message per window; the count is in the summary
            return
        line = f"{iso(now())} [{self.backend}] {msg}"
        with (OUT / "ALERTS.log").open("a") as f:
            f.write(line + "\n")
        self.w["errors"].append(msg)

    def pick_backend(self) -> None:
        if self.a.backend != "auto":
            self.backend = self.a.backend
            return
        for b, u in UNITS.items():
            if unit_state(u).get("ActiveState") in ("active", "activating"):
                self.backend = b
                return
        self.backend = self.backend if self.backend != "auto" else "tf"

    def parse_server(self, lines: list[str]) -> None:
        for ln in lines:
            if RESTART.search(ln):
                self.w["restarts"] += 1
                self.alert("server (re)start seen in log")
                continue
            m = TF_LANES.search(ln)
            if m:
                self.lanes = int(m.group(1))
                self.w["max_lanes"] = max(self.w["max_lanes"], self.lanes)
                continue
            m = TF_DONE.search(ln)
            if m:
                g = m.groupdict()
                rec = {"t": iso(now()), "backend": "tf", "prompt": int(g["prompt"]), "cached": int(g["cached"]),
                       "completion": int(g["tokens"]), "thinking": g["thinking"] == "True", "effort": g["effort"],
                       "finish": g["finish"], "ttft_s": float(g["ttft"]), "decode_tps": float(g["tps"]),
                       "queued_s": float(g["queued"]), "elapsed_s": float(g["elapsed"]),
                       "concurrency": max(1, self.lanes + 1)}
                if g["drafted"]:
                    d, acc = int(g["drafted"]), int(g["acc"])
                    rec.update(mtp_drafted=d, mtp_accepted=acc, mtp_acceptance=round(acc / d, 4) if d else None,
                               mtp_rounds=int(g["rounds"]))
                self.w["requests"].append(rec)
                continue
            m = STRATA_PROMPT.search(ln)
            if m:
                self.pending_prompt = int(m.group("total").replace(",", ""))
                continue
            m = STRATA_DONE.search(ln)
            if m:
                g = m.groupdict()
                self.w["requests"].append({"t": iso(now()), "backend": "strata", "prompt": self.pending_prompt,
                                           "completion": int(g["tokens"]), "decode_tps": float(g["tps"]),
                                           "elapsed_s": float(g["secs"]), "finish": g["finish"].strip(),
                                           "cancel": g["cancel"] == "True", "ttft_s": None})
                self.pending_prompt = None
                continue
            m = HTTP_LINE.search(ln)
            if m:
                s = m.group("status")
                self.w["http"][s] = self.w["http"].get(s, 0) + 1
                if s[0] in "45" and m.group("path").split("?")[0] not in ("/favicon.ico", "/_clamp/stats", "/metrics/prometheus"):  # Prometheus job hotschmoe_dd_strata (Strata-only path)
                    self.alert(f"backend HTTP {s} {m.group('method')} {m.group('path').split('?')[0]}")
                continue
            if BAD.search(ln) and not IGNORE_BAD.search(ln):
                self.alert("server error line: " + re.sub(r"\s+", " ", ln)[:160])

    def parse_frontdoor(self, lines: list[str]) -> None:
        for ln in lines:
            m = HTTP_LINE.search(ln)
            if not m:
                continue
            s, p = m.group("status"), m.group("path").split("?")[0]
            self.w["fd_http"][s] = self.w["fd_http"].get(s, 0) + 1
            if s[0] == "5" or (s[0] == "4" and s not in ("401", "404")):
                if p not in ("/metrics", "/health") or s != "502":
                    self.alert(f"frontdoor HTTP {s} {m.group('method')} {p}")

    def sample(self) -> None:
        logs = [Tail(SERVER_LOG[self.backend], self.state["offsets"]), Tail(FRONTDOOR_LOG, self.state["offsets"])]
        self.parse_server(logs[0].read())
        self.parse_frontdoor(logs[1].read())
        code, body = http_get("/health")
        if code == 200:
            self.w["health_ok"] += 1
            self.health_fail = 0
        else:
            self.w["health_fail"] += 1
            self.health_fail += 1
            if self.health_fail == 2:
                self.alert(f"health failing (status {code}) for 2 consecutive samples")
        if self.backend == "tf" and os.environ.get("SOAK_CLAMP") == "1":  # clamp proxy retired (real sampling)
            c, stats = http_get("/_clamp/stats", 2.0)
            if c == 200 and stats:
                self.w["clamp"] = {k: stats.get(k) for k in ("requests", "temperature_clamped", "dropped", "upstream_errors")}
        t = now()
        for rec in cards():
            name = rec["card"]
            agg = self.w["cards"].setdefault(name, {"temp_pkg_max": 0.0, "temp_vram_max": 0.0, "power_w": [], "act_mhz": [], "throttled": 0})
            agg["temp_pkg_max"] = max(agg["temp_pkg_max"], rec.get("temp_pkg_c", 0))
            agg["temp_vram_max"] = max(agg["temp_vram_max"], rec.get("temp_vram_c", 0))
            if "energy_uj" in rec:
                prev = self.last_energy.get(name)
                if prev and rec["energy_uj"] >= prev[1] and t > prev[0]:
                    agg["power_w"].append((rec["energy_uj"] - prev[1]) / 1e6 / (t - prev[0]))
                self.last_energy[name] = (t, rec["energy_uj"])
            if rec.get("act_mhz"):
                agg["act_mhz"].append(rec["act_mhz"])
            agg["throttled"] += 1 if rec.get("throttle") else 0

    def baseline(self, reqs: list[dict]) -> None:
        b = self.state.setdefault("baseline", {"backend": self.backend, "start": now(), "decode": [], "ttft": [], "done": False})
        if b.get("backend") != self.backend:  # backend switched: start a new baseline hour
            b.clear()
            b.update(backend=self.backend, start=now(), decode=[], ttft=[], done=False)
        if b["done"]:
            return
        b["decode"] += [r["decode_tps"] for r in reqs if r.get("completion", 0) >= 32]
        b["ttft"] += [r["ttft_s"] for r in reqs if r.get("ttft_s") is not None]
        if now() - b["start"] >= 3600 and len(b["decode"]) >= 5:
            b["done"] = True
            b["decode_p50"] = statistics.median(b["decode"])
            b["ttft_p50"] = statistics.median(b["ttft"]) if b["ttft"] else None

    def summarize(self) -> None:
        w, reqs = self.w, self.w["requests"]
        self.baseline(reqs)
        b = self.state["baseline"]
        dec = [r["decode_tps"] for r in reqs if r.get("completion", 0) >= 32]
        ttft = [r["ttft_s"] for r in reqs if r.get("ttft_s") is not None]
        acc = [r["mtp_acceptance"] for r in reqs if r.get("mtp_acceptance") is not None]
        p50d, p50t = (statistics.median(dec) if dec else None), (statistics.median(ttft) if ttft else None)
        us = unit_state(UNITS[self.backend])
        nr = int(us.get("NRestarts", "0") or 0)
        last = self.state.get("nrestarts", {}).get(self.backend)
        if last is not None and nr > last:
            self.alert(f"systemd restarted {UNITS[self.backend]} ({last} -> {nr})")
        self.state.setdefault("nrestarts", {})[self.backend] = nr
        if us.get("ActiveState") not in (None, "active"):
            self.alert(f"{UNITS[self.backend]} is {us.get('ActiveState')}")
        if b.get("done") and len(dec) >= 5 and p50d is not None and p50d < 0.7 * b["decode_p50"]:
            self.alert(f"slowdown: window p50 decode {p50d:.1f} tok/s < 70% of baseline {b['decode_p50']:.1f}")
        if b.get("done") and b.get("ttft_p50") and len(ttft) >= 5 and p50t is not None and p50t > 3 * b["ttft_p50"]:
            self.alert(f"slowdown: window p50 TTFT {p50t:.3f}s > 3x baseline {b['ttft_p50']:.3f}s")
        cards_out = {}
        for k, v in w["cards"].items():
            cards_out[k] = {"temp_pkg_max_c": v["temp_pkg_max"], "temp_vram_max_c": v["temp_vram_max"],
                            "power_w_avg": round(statistics.mean(v["power_w"]), 1) if v["power_w"] else None,
                            "power_w_max": round(max(v["power_w"]), 1) if v["power_w"] else None,
                            "act_mhz_avg": round(statistics.mean(v["act_mhz"])) if v["act_mhz"] else None,
                            "throttled_samples": v["throttled"]}
        summary = {"kind": "window", "start": iso(w["start"]), "end": iso(now()), "backend": self.backend,
                   "requests": len(reqs), "decode_tps_p50": p50d, "decode_tps_p10": pct(dec, 0.1),
                   "ttft_s_p50": p50t, "ttft_s_p90": pct(ttft, 0.9),
                   "prompt_tokens_max": max((r["prompt"] or 0 for r in reqs), default=None),
                   "completion_tokens_sum": sum(r["completion"] for r in reqs),
                   "mtp_acceptance_mean": round(statistics.mean(acc), 4) if acc else None,
                   "max_concurrency": w["max_lanes"], "backend_http": w["http"], "frontdoor_http": w["fd_http"],
                   "health_ok": w["health_ok"], "health_fail": w["health_fail"], "restarts_in_log": w["restarts"],
                   "unit": us, "cards": cards_out, "clamp": w["clamp"], "alerts": len(w["errors"]),
                   "baseline": {k: b.get(k) for k in ("backend", "done", "decode_p50", "ttft_p50")}}
        with (OUT / "soak.jsonl").open("a") as f:
            for r in reqs:
                f.write(json.dumps({"kind": "request", **r}) + "\n")
            f.write(json.dumps(summary) + "\n")
        name = time.strftime("summary-%Y%m%dT%H%M.md", time.gmtime())
        rows = "\n".join(f"| {k} | {v['temp_pkg_max_c']} | {v['temp_vram_max_c']} | {v['power_w_avg']} | {v['power_w_max']} | "
                         f"{v['act_mhz_avg']} | {v['throttled_samples']} |" for k, v in cards_out.items())
        fmt = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
        md = (f"# hotschmoe-dd soak window {summary['start']} -> {summary['end']} ({self.backend})\n\n"
              f"- requests: {len(reqs)}; completion tokens: {summary['completion_tokens_sum']}; max prompt: {summary['prompt_tokens_max']}\n"
              f"- decode tok/s p50 {fmt(p50d)} / p10 {fmt(summary['decode_tps_p10'])}; TTFT s p50 {fmt(p50t, 3)} / p90 {fmt(summary['ttft_s_p90'], 3)}\n"
              f"- MTP acceptance mean: {fmt(summary['mtp_acceptance_mean'], 3)}; max concurrent streams: {w['max_lanes']}\n"
              f"- backend HTTP: {w['http']}; frontdoor HTTP: {w['fd_http']}\n"
              f"- health ok/fail samples: {w['health_ok']}/{w['health_fail']}; restarts in log: {w['restarts']}; unit: {us.get('ActiveState')} NRestarts={nr}\n"
              f"- clamp: {w['clamp']}\n"
              f"- baseline: {summary['baseline']}\n"
              f"- alerts this window: {len(w['errors'])}" + ("".join(f"\n  - {e} (x{w['seen'].get(e, 1)})" for e in w["errors"][:20])) + "\n\n"
              "| card | pkg max C | vram max C | W avg | W max | MHz avg | throttled samples |\n|---|---|---|---|---|---|---|\n" + rows + "\n")
        (OUT / name).write_text(md)
        self.save()
        self.reset_window()

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.state_path)

    def run(self) -> None:
        self.pick_backend()
        next_summary = now() + self.a.summary_s
        while True:
            self.sample()
            self.save()
            if self.a.once:
                self.summarize()
                return
            if now() >= next_summary:
                self.summarize()
                self.pick_backend()
                next_summary += self.a.summary_s
            time.sleep(self.a.sample_s)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["auto", "tf", "strata"], default="auto")
    p.add_argument("--sample-s", type=float, default=60)
    p.add_argument("--summary-s", type=float, default=900)
    p.add_argument("--once", action="store_true", help="one sample + one summary, then exit")
    Monitor(p.parse_args()).run()
