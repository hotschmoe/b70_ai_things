#!/usr/bin/env python3
"""Prepare an unapplied, source-pinned candidate. No GPU/image modification."""
import argparse
import difflib
import hashlib
from pathlib import Path
MODEL_SHA='fb15e26fcfe4f7ab72a61a64fee8b8ea0c49b7f28a23b6a9dede099e41489231'
WORKER_SHA='936d65f7e9e85e0c67ada8210ce26108cf7868c4f991f76d0ad3bc125c07dc3a'
MODEL='vllm/model_executor/layers/mamba/mamba_utils.py'
WORKER='vllm/v1/worker/mamba_utils.py'
ADD='''

def get_xpu_gdn_conv_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Pinned legacy XPU GDN: each speculative slot holds three valid rows."""
    assert not is_conv_state_dim_first(), "XPU GDN adapter requires SD layout"
    assert state.dim() == 3 and state.size(1) >= 3
    assert num_accepted_tokens >= 1
    src = state[block_ids[cur_block_idx + num_accepted_tokens - 1], :3]
    return MambaCopySpec(start_addr=src.data_ptr(), num_elements=src.numel())


@functools.lru_cache
 def_placeholder
'''.replace(' def_placeholder','def _use_xpu_gdn_prefix_conv_copy() -> bool:\n    import os\n    enabled = os.environ.get("B70_XPU_GDN_PREFIX_CONV_COPY", "0")\n    if enabled == "0":\n        return False\n    if enabled != "1":\n        raise ValueError("B70_XPU_GDN_PREFIX_CONV_COPY must be 0 or 1")\n    from vllm.platforms import current_platform\n    if not current_platform.is_xpu():\n        raise RuntimeError("XPU GDN adapter requested on a non-XPU platform")\n    import hashlib\n    from pathlib import Path\n    import vllm_xpu_kernels._xpu_C as native\n    expected = "271db0d4882124e21ac6a4d080bfeab303fbb08b9ec10e11f21d10fb0723998f"\n    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != expected:\n        raise RuntimeError("XPU GDN adapter native identity mismatch")\n    if is_conv_state_dim_first():\n        raise RuntimeError("XPU GDN adapter requires SD layout")\n    return True\n')

def changes(model,worker):
    assert hashlib.sha256(model.encode()).hexdigest()==MODEL_SHA
    assert hashlib.sha256(worker.encode()).hexdigest()==WORKER_SHA
    newmodel=model.replace('\n\ndef get_temporal_copy_spec(',ADD+'\n\ndef get_temporal_copy_spec(',1)
    old='    def gated_delta_net_state_copy_func(cls):\n        return (get_conv_copy_spec, get_temporal_copy_spec)'
    new='    def gated_delta_net_state_copy_func(cls):\n        conv_copy = (get_xpu_gdn_conv_copy_spec if _use_xpu_gdn_prefix_conv_copy()\n                     else get_conv_copy_spec)\n        return (conv_copy, get_temporal_copy_spec)'
    assert old in newmodel;newmodel=newmodel.replace(old,new,1)
    newworker=worker.replace('    get_conv_copy_spec,\n','    get_conv_copy_spec,\n    get_xpu_gdn_conv_copy_spec,\n',1)
    newworker=newworker.replace('                        copy_func is get_conv_copy_spec\n','                        copy_func is get_conv_copy_spec\n                        or copy_func is get_xpu_gdn_conv_copy_spec\n',1)
    old='                    if copy_func is get_conv_copy_spec:\n'
    new='''                    if copy_func is get_xpu_gdn_conv_copy_spec:
                        # Explicit legacy-GDN copy marker. Zero width selects
                        # the generic flat-checkpoint branch; the shared
                        # temporal byte-copy route selects src_col + token_bias.
                        # Only three rows are valid in each native prefix slot.
                        assert not is_conv_state_dim_first()
                        assert state.dim() == 3 and state.size(1) >= 3
                        self.state_conv_widths[idx] = 0
                        self.state_inner_sizes[idx] = 3 * state.stride(1)
                    elif copy_func is get_conv_copy_spec:
'''
    assert old in newworker;newworker=newworker.replace(old,new,1)
    return newmodel,newworker


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-source',type=Path,required=True);p.add_argument('--worker-source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    old=[a.model_source.read_text(),a.worker_source.read_text()];new=changes(*old);patch=''
    for name,before,after in zip([MODEL,WORKER],old,new):
        compile(after,name,'exec');path=a.out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(after);patch+=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (a.out/'xpu-gdn-prefix-conv-copy.patch').write_text(patch)
    print(a.out)
if __name__=='__main__':main()
