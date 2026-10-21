"""L1: stock vLLM CLI plus X5's read-only /server_info config route (no env, credentials redacted)."""
import sys

sys.path.insert(0, '/mnt/vm_8tb/b70/build/vllm-flashnext/lmx-l1/frozen-repo/vllm/flashnext/harness')

if __name__ == '__main__':
    from x5_server_info import install

    install()
    from vllm.entrypoints.cli.main import main

    main()
