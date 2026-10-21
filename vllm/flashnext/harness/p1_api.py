"""Opt-in P1 bucket overlay plus stock CLI and benchmark metadata endpoint."""
# Top-level install is intentional: multiprocessing spawn reexecutes this file
# as __mp_main__ before the EngineCore target runs. No GPU work happens here.
from b70_p1_xpu.serving import install as install_p1
install_p1()

if __name__ == '__main__':
    from x5_server_info import install
    install()
    from vllm.entrypoints.cli.main import main
    main()
