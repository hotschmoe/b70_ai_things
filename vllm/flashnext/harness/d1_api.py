"""Stock vLLM CLI plus the existing read-only benchmark config endpoint."""

from x5_server_info import install


if __name__ == "__main__":
    install()
    from vllm.entrypoints.cli.main import main

    main()
