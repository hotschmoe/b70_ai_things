"""Parse vLLM Prometheus text for prefix-cache hit counters."""

from __future__ import annotations


def scrape_counter(
    text: str,
    name: str,
    label: str | None = None,
    default: float | None = None,
) -> float:
    """Sum samples of ``name`` (with or without the ``_total`` suffix).

    ``label`` matches a raw label fragment such as ``source="host"``.
    ``_created`` lines are ignored. A series Prometheus has not emitted yet
    returns ``default`` when that is set (a host tier stays absent until the
    first external hit).
    """
    total = 0.0
    found = False
    prefixes = (name + "{", name + " ", name + "_total{", name + "_total ")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        head = line.split()[0]
        if head.endswith("_created") or "_created{" in head:
            continue
        if not head.startswith(prefixes):
            continue
        if label is not None and label not in head:
            continue
        total += float(line.split()[-1])
        found = True
    if not found:
        if default is not None:
            return default
        raise KeyError(name if label is None else f"{name} {label}")
    return total


def counter_delta(before: str, after: str, name: str, label: str | None = None) -> float:
    """Difference of two scrapes. A missing series counts as zero."""
    return scrape_counter(after, name, label, default=0.0) - scrape_counter(
        before, name, label, default=0.0
    )
