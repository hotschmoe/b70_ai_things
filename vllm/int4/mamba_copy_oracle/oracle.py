#!/usr/bin/env python3
"""Plan by default. --run-xpu requires an external selected-card lease lifecycle."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

WORKER_SHA = '936d65f7e9e85e0c67ada8210ce26108cf7868c4f991f76d0ad3bc125c07dc3a'
MODEL_SHA = 'fb15e26fcfe4f7ab72a61a64fee8b8ea0c49b7f28a23b6a9dede099e41489231'
BLOCK_IDS = [3, 5, 1, 6, 2, 7, 0, 4]
PADDED_STRIDE = 832 * 4096


def cases():
    rows = [dict(mtp=3, history=6, src=src, dst=dst, bias=bias, padded=True)
            for src, dst in [(0, 7), (4, 0)] for bias in [0, 1, 3]]
    rows += [dict(mtp=3, history=6, src=src, dst=dst, bias=0, padded=True)
             for src, dst in [(-1, 7), (0, 0)]]
    rows += [dict(mtp=0, history=3, src=src, dst=dst, bias=0, padded=True)
             for src, dst in [(0, 7), (4, 0)]]
    rows += [dict(mtp=mtp, history=history, src=0, dst=7, bias=0, padded=False)
             for mtp, history in [(0, 3), (3, 6)]]
    return rows


def geometry(case):
    conv = case['history'] * 10240 * 2
    temporal = 48 * 128 * 128 * 4
    return dict(conv_bytes=conv, temporal_bytes=temporal,
                page_bytes=PADDED_STRIDE if case['padded'] else conv + temporal)


def regions(case):
    g = geometry(case)
    if case['src'] < 0 or case['src'] == case['dst']:
        return []
    stride = g['page_bytes']
    source = BLOCK_IDS[case['src']]
    dest = BLOCK_IDS[case['dst']]
    temporal_source = BLOCK_IDS[case['src'] + case['bias']]
    return [(source*stride + case['bias']*10240*2, dest*stride,
             (case['history']-case['bias'])*10240*2),
            (temporal_source*stride+g['conv_bytes'], dest*stride+g['conv_bytes'],
             g['temporal_bytes'])]


def run():
    import torch
    from vllm.v1.worker import mamba_utils as worker
    from vllm.model_executor.layers.mamba import mamba_utils as model
    torch.set_num_threads(2)
    assert hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest() == WORKER_SHA
    assert hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest() == MODEL_SHA
    assert not model.is_conv_state_dim_first(), 'oracle covers actual SD control only'
    for kernel in [worker.precopy_mamba_align_fused_kernel, worker.batch_memcpy_kernel]:
        assert Path(inspect.getsourcefile(kernel.fn)).resolve() == Path(worker.__file__).resolve()
    results = []
    for index, case in enumerate(cases()):
        g = geometry(case); page = g['page_bytes']
        generator = torch.Generator(device='cpu').manual_seed(9010+index)
        before = torch.randint(0, 256, (8*page,), dtype=torch.uint8, generator=generator)
        expected = before.clone()
        for src, dst, size in regions(case):
            expected[dst:dst+size] = before[src:src+size]
        raw = before.to('xpu')
        # Exact allocator: adjacent conv/SSM within each page; padding follows.
        pages = raw.view(8, page)
        conv = pages[:, :g['conv_bytes']].view(torch.float16).view(8, case['history'], 10240)
        temporal = pages[:, g['conv_bytes']:g['conv_bytes']+g['temporal_bytes']].view(torch.float32).view(8, 48, 128, 128)
        assert conv.stride(0)*2 == temporal.stride(0)*4 == page
        assert temporal.data_ptr()-conv.data_ptr() == g['conv_bytes']
        if case['padded']:
            assert not conv.is_contiguous() and not temporal.is_contiguous()
        bt = torch.tensor([BLOCK_IDS], dtype=torch.int32, device='xpu')
        dtypes = dict(state_base_addrs=torch.int64, state_block_strides=torch.int64,
                      state_elem_sizes=torch.int32, state_inner_sizes=torch.int64,
                      state_conv_widths=torch.int32, state_group_indices=torch.int32,
                      state_dim_row_count=torch.int32, state_dim_row_stride=torch.int64)
        ctx = NS(**{key:torch.zeros(2, dtype=dtype, device='xpu') for key,dtype in dtypes.items()},
                 mamba_group_ids=[0], num_groups=1, num_layers=1, num_state_types=2,
                 is_initialized=False, block_table_ptrs=torch.zeros(1,dtype=torch.int64,device='xpu'))
        worker.MambaSpecDecodeGPUContext.initialize_from_forward_context(ctx,
            NS(kv_cache_groups=[NS(layer_names=['fixture'])]),
            {'fixture':NS(kv_cache=[conv,temporal])},
            (model.get_conv_copy_spec,model.get_temporal_copy_spec),[bt])
        metadata = {key:getattr(ctx,key).cpu().tolist() for key in dtypes}
        assert metadata['state_block_strides'] == [page,page]
        assert metadata['state_inner_sizes'] == [10240,48*128*128]
        tensor = lambda value:torch.tensor([value],dtype=torch.int32,device='xpu')
        if case['mtp']:
            worker.MambaSpecDecodeGPUContext.run_fused_precopy(ctx,1,
                tensor(case['dst']),tensor(case['src']),tensor(case['bias']),None)
            route = 'actual run_fused_precopy -> precopy_mamba_align_fused_kernel'
        else:
            specs = [fn(state,BLOCK_IDS,case['src'],case['bias']+1) for state,fn in
                     [(conv,model.get_conv_copy_spec),(temporal,model.get_temporal_copy_spec)]]
            source_ptrs=torch.tensor([s.start_addr for s in specs],dtype=torch.uint64,device='xpu')
            dest_ptrs=torch.tensor([state[BLOCK_IDS[case['dst']]].data_ptr() for state in
                                   [conv,temporal]],dtype=torch.uint64,device='xpu')
            sizes=torch.tensor([spec.num_elements*state.element_size() for spec,state in
                                zip(specs,[conv,temporal])],dtype=torch.int32,device='xpu')
            worker.batch_memcpy(source_ptrs,dest_ptrs,sizes)
            route = 'actual copy specs -> batch_memcpy -> batch_memcpy_kernel'
        torch.xpu.synchronize()
        actual = raw.cpu()
        mismatch = int((actual != expected).sum())
        rows = regions(case)
        source_untouched = all(torch.equal(actual[src:src+size],before[src:src+size])
                               for src,dst,size in rows)
        row = dict(case=case,geometry=g,route=route,metadata=metadata,
                   mismatched_bytes=mismatch,source_untouched=source_untouched,
                   all_untargeted_bytes_covered=True,passed=mismatch==0 and source_untouched)
        results.append(row)
        print(json.dumps(dict(event='case',index=index,**row)),flush=True)
    result=dict(worker_source_sha256=WORKER_SHA,model_source_sha256=MODEL_SHA,
                execution='actual installed Triton copy kernels',cases=results,
                passed=all(r['passed'] for r in results),
                scope='Synthetic padded Mamba block copy only; no scheduler/publication/native GDN correctness claim')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['passed'] else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-xpu',action='store_true')
    args=parser.parse_args()
    if not args.run_xpu:
        print(json.dumps(dict(status='PREPARED_NOT_RUN',cases=[dict(**c,geometry=geometry(c),regions=regions(c)) for c in cases()],
            requirement='External gpu-run --card N lifecycle, device pin, strict pre/post health, bounded container; no single-lease reset'),indent=2))
        return 0
    return run()


if __name__ == '__main__':
    raise SystemExit(main())
