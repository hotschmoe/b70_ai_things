"""Identical default-off native-call observer in API and spawned workers."""
from branch_hook import install
install()
if __name__=='__main__':
    from x5_server_info import install as install_metadata
    install_metadata()
    from vllm.entrypoints.cli.main import main
    main()
