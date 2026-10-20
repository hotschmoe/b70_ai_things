"""b70 GDN decode and paged attention ops. Load explicitly; nothing imports this at startup."""

from b70_gdn_xpu.ops import register

__all__ = ["register"]
