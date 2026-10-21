"""Exercise the actual paged QsaRunner, including 3-row text positions.

One physical card, pinned by xpu-run. No model weights. Warm every runner
before capture, then compare eager and replay with changed positions/pages.
"""
from types import SimpleNamespace as NS
import signal

import torch
from b70_qsa_xpu.layout import page_layout
from b70_qsa_xpu.model_hook import QsaRunner


class Linear(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.op = torch.nn.Linear(n_in, n_out, bias=False, dtype=torch.bfloat16, device="xpu")

    def forward(self, x):
        return self.op(x), None


class Attention:
    def __init__(self):
        self.config = NS(rope_parameters={"rope_theta": 1e7, "partial_rotary_factor": 0.25},
                         indexer_compress_ratio=4, indexer_budget=2048, rms_norm_eps=1e-6)
        self.num_heads, self.num_kv_heads, self.head_dim = 24, 2, 256
        self.rotary_emb = NS(rotary_dim=64)
        self.qkv_proj = Linear(128, 24 * 512 + 2 * 512)
        self.o_proj = Linear(24 * 256, 128)
        self.indexer = NS(index_n_heads=4, index_head_dim=128, index_kv_heads=1,
                          q_layernorm=NS(weight=torch.zeros(128, device="xpu")),
                          k_layernorm=NS(weight=torch.zeros(128, device="xpu")),
                          index_qk_proj=Linear(128, 640))
        page = page_layout(2, 256, 128, 4, 64, pooled=True)["page_bytes"]
        self.attn = NS(layer_name="layers.3.self_attn.attn",
                       kv_cache=torch.zeros(520, 1, 1, page, device="xpu", dtype=torch.uint8))

    def _project_qkv_gate(self, x, positions):
        del positions
        qg, k, v = x.split([12288, 512, 512], dim=-1)
        q, gate = qg.reshape(-1, 24, 512).chunk(2, dim=-1)
        return q.reshape(-1, 6144), k, v, gate.reshape(-1, 6144)


@torch.inference_mode()
def main():
    signal.alarm(120)
    torch.manual_seed(61)
    attn = Attention()
    runner = QsaRunner(attn, attn.indexer)
    runner._limits = lambda: (64, 4, 128)
    meta = NS(max_query_len=1, num_reqs=4,
              block_table=torch.arange(1, 513, device="xpu", dtype=torch.int32).reshape(4, 128),
              slot_mapping=torch.arange(4, device="xpu", dtype=torch.int64))
    runner._metadata = lambda: meta
    hidden = torch.randn(4, 128, dtype=torch.bfloat16, device="xpu")
    positions = torch.zeros(3, 4, dtype=torch.int64, device="xpu")
    # This first real forward binds pages, builds tables and loads all ops.
    runner.forward(hidden, positions)
    torch.xpu.synchronize()
    batch = runner.batch
    assert batch is not None and runner.buf is None
    for rows in (1, 2, 4):
        meta.slot_mapping.fill_(-1)
        meta.slot_mapping[:rows] = torch.arange(rows, device="xpu")
        runner.forward(hidden[:rows], positions[:, :rows])
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            captured = runner.forward(hidden[:rows], positions[:, :rows])
        for pos in (0, 1, 4, 2052, 4099):
            positions.fill_(pos)
            # Both paths start with the same explicit durable/cache state.
            state = [(t, t.clone()) for t in (
                batch.pages, runner.side_tail, runner.side_dead,
                runner.side_block_pos, runner.side_length)]
            eager = runner.forward(hidden[:rows], positions[:, :rows]).clone()
            torch.xpu.synchronize()
            for tensor, saved in state:
                tensor.copy_(saved)
            graph.replay()
            torch.xpu.synchronize()
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
            print(f"X6_QSA_GRAPH rows={rows} pos={pos} exact=1", flush=True)
        del graph
    print("X6_QSA_RUNNER_GRAPH_OK", flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
