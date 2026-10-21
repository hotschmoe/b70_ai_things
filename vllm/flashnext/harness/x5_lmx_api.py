"""Research API with a redacted, read-only live config proof route."""
from x5_api import bootstrap


if __name__ == '__main__':
    bootstrap()
    from x5_server_info import install

    install()
    from vllm.entrypoints.cli.main import main

    main()
