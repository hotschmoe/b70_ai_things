# ---- partial reference: only the first feature groups (no aggregates/grouping/joins/UPDATE/DELETE) ----
_orig_core = Database.core
_orig_write = Database.write


def _core(self, c, params, outer):
    if c["joins"] or c["group"] or c["having"] is not None or any(it[0] == "e" and has_agg(it[1]) for it in c["items"]):
        raise SQLError("syntax", "partial reference")
    return _orig_core(self, c, params, outer)


def _write(self, node, params):
    if node[0] in ("update", "delete"):
        raise SQLError("syntax", "partial reference")
    return _orig_write(self, node, params)


Database.core = _core
Database.write = _write
