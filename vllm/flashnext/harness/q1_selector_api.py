"""Opt-in selector hook in API and spawned worker, plus config metadata."""
from q1_selector_hook import install as install_selector

install_selector()

if __name__ == '__main__':
    from x5_server_info import install
    install()
    from vllm.entrypoints.cli.main import main
    main()
