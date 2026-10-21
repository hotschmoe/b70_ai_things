"""Publish actual runtime configuration using the existing research helper."""
from x5_server_info import install
install()
if __name__ == '__main__':
    from vllm.entrypoints.cli.main import main
    main()
