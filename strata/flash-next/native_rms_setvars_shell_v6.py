"""Source-only shell guard and root-only CPU/no-device image smoke recipes."""
PREFIX='set -e; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; '
def smoke_recipes(compiler_image,runtime_image):
 common=['docker','run','--rm','--network','none','--cpus','2','--memory','512m','--memory-swap','512m','--pids-limit','64','--user','1000:1000','--entrypoint','/bin/bash']
 return [common+[compiler_image,'-c',PREFIX+'exec /opt/intel/oneapi/compiler/2026.1/bin/icpx --version'],common+[runtime_image,'-c',PREFIX+'exec /opt/b70-c1-python/bin/python --version']]
