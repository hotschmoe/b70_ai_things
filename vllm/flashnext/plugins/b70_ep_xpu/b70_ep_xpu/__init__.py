"""EP sidecar: vLLM stays on card 0, this package owns card 1."""

from .ops import load


def register() -> None:
    """Load the transport library, patch the GGUF MoE method, install graph hooks.

    PLE and QSA hooks run before the model is built. B70_QSA unset leaves
    the 35B path alone. The PLE guard is a no-op until a host table is installed.
    """
    load()
    from .hooks import install as install_hooks
    from .method import install as install_method
    from .ple_guard import install as install_ple
    from .qsa_cover import install as install_qsa

    install_ple()
    install_qsa()
    install_method()
    install_hooks()
