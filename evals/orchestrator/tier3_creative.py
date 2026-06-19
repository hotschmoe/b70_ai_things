"""Tier 3 — curated creative/visual builds. Semi-objective.

For each prompt: generate one self-contained HTML file, render it headless, and record the OBJECTIVE
sub-signal: did it load with zero console/page errors and a non-blank body ("renders_clean"). We also
save the raw HTML + a screenshot so you (or a vision-LLM later) can do PAIRWISE A/B vs the bf16
reference, position-swapped (README §3/§6).

Generation uses the CHAT endpoint (creative front-end work is a chat task). Note: chat-template
differences across quants must be held constant — they're served by the same vLLM, so the template is
identical here; the only variable is the weights. Keep it that way.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from common import RunContext, write_json

_FENCE = re.compile(r"```(?:html)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_html(text: str) -> str:
    m = _FENCE.search(text)
    if m and ("<" in m.group(1)):
        return m.group(1).strip()
    t = text.strip()
    if t.lower().startswith(("<!doctype", "<html")):
        return t
    # last resort: wrap whatever came back so it at least renders
    return f"<!DOCTYPE html><html><body>{t}</body></html>"


def _render(html_path: Path, view: dict, delay_ms: int) -> dict:
    """Render with Playwright; return {renders_clean, console_errors, screenshot, non_blank}."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"skipped": True, "error": "playwright missing: pip install playwright && playwright install chromium"}
    errors: list[str] = []
    shot = html_path.with_suffix(".png")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": view.get("width", 1024), "height": view.get("height", 768)})
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.goto(html_path.as_uri())
        page.wait_for_timeout(delay_ms)
        non_blank = page.evaluate("() => document.body && document.body.innerText.length + document.querySelectorAll('canvas,svg,img').length > 0")
        page.screenshot(path=str(shot))
        browser.close()
    return {"renders_clean": len(errors) == 0, "console_errors": errors,
            "non_blank": bool(non_blank), "screenshot": str(shot)}


def run(ctx: RunContext, suite_path: str, limit: int | None = None) -> dict:
    from common import make_client
    client = make_client(ctx.endpoint)
    suite = yaml.safe_load(Path(suite_path).read_text())
    defaults = suite.get("defaults", {})
    system = defaults.get("system", "")
    rd = defaults.get("render", {})
    view = rd.get("viewport", {})
    delay = rd.get("screenshot_delay_ms", 2000)
    out_dir = ctx.out_dir / "tier3_creative"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = suite.get("prompts", [])
    if limit:
        prompts = prompts[:limit]
    items = []
    clean = 0
    for pr in prompts:
        pid = pr["id"]
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": pr["prompt"]}]
        resp = client.chat.completions.create(
            model=ctx.model_id, messages=msgs,
            temperature=ctx.sampling.get("temperature", 0.0),
            seed=ctx.sampling.get("seed", 1234),
            max_tokens=ctx.sampling.get("max_tokens", 4096),
        )
        text = resp.choices[0].message.content or ""
        html = extract_html(text)
        html_path = out_dir / f"{pid}.html"
        html_path.write_text(html)
        (out_dir / f"{pid}.raw.txt").write_text(text)
        render = _render(html_path, view, delay)
        if render.get("renders_clean") and render.get("non_blank"):
            clean += 1
        items.append({"id": pid, "tags": pr.get("tags", []),
                      "chars": len(html), "render": render})
        print(f"[tier3] {pid}: clean={render.get('renders_clean')} non_blank={render.get('non_blank')}")

    result = {"tier": 3, "suite": suite.get("suite"), "n": len(items),
              "renders_clean_count": clean,
              "renders_clean_rate": (clean / len(items)) if items else None,
              "items": items,
              "note": "Subjective win-rate vs bf16 is a SEPARATE pairwise pass over the saved screenshots."}
    write_json(ctx.out_dir / "tier3_creative.json", result)
    print(f"[tier3] renders-clean {clean}/{len(items)}")
    return result
