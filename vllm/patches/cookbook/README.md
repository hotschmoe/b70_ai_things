# Cookbook MTP patches (ported)

Ported from [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
(MIT) for our dual-B70 lab. Applied at container start by
`vllm/cookbook_campaign/launch.sh`.

| File | When | Purpose |
|------|------|---------|
| `patch_mtp_nightly.py` | Public `vllm/vllm-openai-xpu` 0.26.1rc* | Force BF16 MTP draft when `B70_MTP_BF16_DRAFT=1` (checkpoint lacks `-:mtp` dynamic exclude) |
| `patch_mtp_bf16_draft_v0260.py` | Our `vllm-xpu-env:v0260` (0.26.0) | Same intent; different source anchors |
| `patch_mtp_boundary.py` | Both | Exact max-model-len partial final MTP group -> non-spec prefill (XPU GDN needs full groups) |
| `patch_gdn_mixed_split_v5.py` | f01e24f6 0.27 (best-effort) | Split mixed spec/non-spec GDN batches. C1 homogeneous is a no-op. |
| `patch_draft_lmhead_int4.py` | f01e24f6 / 2c427ef | Runtime RTN INT4 of the draft LM head. Env `B70_DRAFT_LMHEAD_INT4=1`. |
| `patch_draft_mtp_int4.py` | f01e24f6 / 2c427ef | Runtime RTN INT4 of 5 MTP dense linears. Env `B70_DRAFT_MTP_INT4=1`. |
| `apply_mtp_patches.py` | Both | Auto-select draft patch + boundary + 2026.08.19 overlay |

## Usage

```bash
# inside container, before serve:
export B70_MTP_BF16_DRAFT=1
python /patches/apply_mtp_patches.py
```

Or via the campaign launcher:

```bash
bash vllm/cookbook_campaign/launch.sh dense27 mtp4 on 8000
```
