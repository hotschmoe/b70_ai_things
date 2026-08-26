# Evaluation Harness

Use `configs/models.yaml` as the active model identity registry. Before every
graded run, query `/v1/models` and require an exact served-model-id match.

The retained harness supports deterministic distribution, code, reasoning, and
creative comparisons. Create `evals/.venv` from `requirements.txt` after the
backend refresh; the prior environment and raw result history are quarantined.

Keep generated code execution inside the network-disabled sandbox. Record every
graded run in `JOURNAL.md` as config, command, result, and verdict.
