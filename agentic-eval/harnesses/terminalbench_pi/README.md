# Pi + Terminal-Bench local harness

This harness runs the same Pi coding agent against local OpenAI-compatible model
endpoints inside Harbor's Docker task environments. Terminal-Bench 3.0 is the
headline dataset. The primary comparison is cumulative verifier-passing tasks over
wall time, not model token throughput.

Pinned components:

- Harbor 0.22.0
- Terminal-Bench 3.0, package content hash
  `sha256:a32a61879ea94eb9dc16fa1fbeb398759f0c07ca633d9d1f6aec760207036da3`
  (74 upstream tasks; ordered names in `tasks-3.0.0.txt`)
- Pi 0.84.3
- concurrency: 1 for the product-choice headline

The reproducible local score uses 70 tasks. It excludes the four H100-only
tasks (`exam-pdf-eval`, `fp8-rmsnorm-gemm`, `jax-speedrun-gpu`, and
`math-eval-grader`) because the two B70s are occupied by the inference endpoint
and are not CUDA H100s. These are exclusions, not model failures.

Install the runner:

```bash
bash agentic-eval/harnesses/terminalbench_pi/setup.sh
```

The full run contract and exact three serving arms are documented in
`docs/20260824_pi_terminalbench_model_selection.md`.
