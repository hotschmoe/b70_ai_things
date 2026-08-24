#!/usr/bin/env bash
# Exact endpoint identities for the three-arm product-choice campaign.
set -uo pipefail

TB_ARM_LIST=(qwen36-w8a8 qwen38-xl ornith15-w8a8)

tb_set_arm() {
  case "${1:-}" in
    qwen36-w8a8)
      TB_ARM=qwen36-w8a8
      TB_SERVED=qwen36-27b-w8a8-mtp
      TB_BACKEND=sglang
      TB_SCHEME=W8A8-sqgptq
      TB_CONTEXT=131072
      ;;
    qwen38-xl)
      TB_ARM=qwen38-xl
      TB_SERVED=hotschmoe-dd
      TB_BACKEND=llamacpp
      TB_SCHEME=UD-Q4_K_XL-unsloth
      TB_CONTEXT=262144
      ;;
    ornith15-w8a8)
      TB_ARM=ornith15-w8a8
      TB_SERVED=ornith-1.5-35b-a3b-W8A8-rtn-mtp-shisa
      TB_BACKEND=sglang
      TB_SCHEME=W8A8-rtn-mtp-shisa
      TB_CONTEXT=262144
      ;;
    *)
      echo "unknown arm '${1:-}'; valid: ${TB_ARM_LIST[*]}" >&2
      return 2
      ;;
  esac
  export TB_ARM TB_SERVED TB_BACKEND TB_SCHEME TB_CONTEXT
}
