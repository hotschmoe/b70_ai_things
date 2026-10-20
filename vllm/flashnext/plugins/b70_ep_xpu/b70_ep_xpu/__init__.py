"""EP sidecar: vLLM stays on card 0, this package owns card 1."""

from .ops import load


def register() -> None:
    """Load the transport library, patch the GGUF MoE method, install graph hooks."""
    load()
    from .hooks import install as install_hooks
    from .method import install as install_method

    install_method()
    install_hooks()
