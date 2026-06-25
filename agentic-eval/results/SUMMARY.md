# Agentic-eval scoreboard (generated)

## Scoreboard

| config | arch | Aider (codegen control) | BFCL multi-turn (tool isolator) | tau2 (multi-turn tool) | SWE/mini (agentic coding) | total wall | total tokens | gen tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `27b-int4` | dense | 40.0% (pass_rate_2 n=5) | -- | -- | 0.0% (resolved n=3) | 109m43s | 305,082 | 28.9 |
| `27b-w8a8` | dense | 40.0% (pass_rate_2 n=5) | -- | -- | 33.3% (resolved n=3) | 139m28s | 1,068,849 | 13.3 |
| `35b-int4` | moe | 0.0% (pass_rate_2 n=5) | -- | -- | 0.0% (resolved n=3) | 27m48s | 811,410 | 58.8 |

### Within-architecture quant deltas (int4 - w8a8)

- dense aider: int4 40.0% - w8a8 40.0% = **+0.0 pp**  (int4 better)
- dense swe: int4 0.0% - w8a8 33.3% = **-33.3 pp**  (int4 WORSE)

_Greedy (temp=0). Scores are concurrency-invariant; wall-clock/tokens are at the fixed eval concurrency. Read int4-vs-w8a8 within an arch; do not read dense-vs-moe as a quant effect._
