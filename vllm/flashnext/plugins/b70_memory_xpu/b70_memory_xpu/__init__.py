"""Small, opt-in GGUF memory fixes. Importing this module has no GPU effects."""
import os

_INSTALLED = False


def register():
    global _INSTALLED
    if _INSTALLED:
        return
    fix = os.environ.get("B70_MEMORY_FIX") == "1"
    report = os.environ.get("B70_MEMORY_REPORT")
    if not fix and not report:
        return
    _INSTALLED = True
    if fix:
        from .fixes import install
        install()
        from .allocator import install as install_allocator
        install_allocator()
    if report:
        from .observe import install
        install(report)
