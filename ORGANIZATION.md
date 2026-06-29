# ORGANIZATION.md -- repo layout + the anti-clobbering contract

> UPDATE 2026-06-29: the repo is now SPLIT BY BACKEND. The authoritative layout + shelf
> rules live in `AGENTS.md` (Repo Layout Contract + Shelf rules). Key deltas from the
> tiers below: backend-specific code lives under `sglang/` and `vllm/` roots; shared custom
> kernel SOURCE lives in `kernels/` (built per-backend); the shelf is
> `rdy_to_serve/<backend>/<model-quant>/` with EXACTLY ONE best config per (backend, model,
> quant) -- no variations; model weights live in `models/files/` (git-ignored). Read every
> `rdy_to_serve/<m>/` reference below as `rdy_to_serve/<backend>/<m>/`. The mutability-tier
> reasoning here still holds; this doc is pending a full rewrite in the docs pass.

Why this exists: we keep clobbering working serve configs, patches, and docker images while
testing new stuff. `rdy_to_serve/` was the first fix (a "release shelf" separate from the
lab notebook). This doc generalizes that into a layout + a set of mutability contracts so a
known-good serve never silently breaks.

ASCII only (CLAUDE.md style rule). Use `->` not arrows, `...` not ellipsis.


## The core idea: separate tiers by MUTABILITY CONTRACT, not by topic

The bug is that throwaway experiments, depended-on tools, and shipped recipes all live in one
namespace (`scripts/NN_*.sh` + root), so editing one to "test new stuff" clobbers another's
production. Fix = four tiers, each with an explicit contract for who may change it and how.

```
  TIER            DIR                 CONTRACT
  --------------  ------------------  ----------------------------------------------------
  LAB NOTEBOOK    scripts/NN_*.sh     append-only. NEVER edit a committed NN to test new
                                      stuff -- copy to the next number. Evidence, not infra.
  TOOLS           bin/                gpu-run, the generic serve engine, 35_sweep_bench, the
                                      DP wrapper. STABLE: depended on by lab + daily driver
                                      + golden path. Change only behind the SWEEP GATE.
  IMAGE RECIPES   images/<tag>/       a Dockerfile (+ recorded digest) per image tag. Tags
                                      are IMMUTABLE: new build -> new tag, never overwrite.
  GOLDEN PATH     rdy_to_serve/<m>/   the current-best, reproducible serve for one config.
                                      Self-contained: serve.sh + patches/ + README, pins the
                                      image DIGEST it was verified against. Change behind the
                                      SWEEP GATE.
```

Mental model:
```
  LAB NOTEBOOK  ->  graduates a recipe to  ->  GOLDEN PATH
  (mutable history)                            (frozen, reproducible)
  the NN script that birthed a recipe is then FROZEN HISTORY, not the thing you run.
```


## Tier detail

### scripts/  (lab notebook)
- Append-only. The numbered chronology is the value; do not rewrite it.
- HARD RULE: never edit a committed `NN_*.sh` to test a new idea. Copy to the next number.
- Anything in here that is actually re-run constantly is mis-filed -> it belongs in `bin/`.

### bin/  (tools -- NEW)
- Move the genuinely-infrastructural scripts OUT of the numbered namespace:
  `gpu-run`, `30_serve_w4a8_graph.sh` (the generic serve engine), `35_sweep_bench.sh`,
  `64_dataparallel_2rep.sh` (DP wrapper). These have a STABILITY contract.
- Once tools are here, "never edit a committed NN" can finally hold (the things you
  legitimately re-edit are no longer NN scripts).
- Biggest blast radius in the repo (lab + daily driver + golden path all call these) ->
  strictest gate. Changes go through the SWEEP GATE below.

### images/  (image recipes -- NEW; this is the real reproducibility fix)
- TODAY images are built by `docker commit` (scripts/47 etc.) onto MUTABLE tags, with no
  Dockerfile. `rdy_to_serve` pins a tag STRING -- but a tag is a mutable pointer, so
  "reproducible" is currently false: a rebuild/prune/base-shift silently changes it.
- The correct way:
  1. REPRODUCIBLE BUILDS, not snapshots. A `Dockerfile` per tag under `images/<tag>/`,
     `FROM` a base pinned by DIGEST (`@sha256:...`), `COPY` patch files in. Image = pure
     function of (base digest + repo state).
  2. IMMUTABLE TAGS. Never overwrite `:int8`. New build -> new tag (date or git-sha
     suffixed). Record the resulting digest.
  3. PRISTINE BASE. `vllm-xpu-base:0.23.0` NEVER carries a behavior-changing patch. This
     is what structurally stops "a MoE patch breaks dense models."
- PATCH DECISION RULE (answers: will a custom patch for one model class break another?):
  ```
    pure-Python patch (no compile)?
      YES -> BIND-MOUNT from the model dir at runtime (-v patch:...:ro).
             Blast radius = ONE container, ONE model. A MoE patch cannot reach a dense
             model -- it is not mounted there. (rdy_to_serve already does this for quark.py.)
      NO  -> needs compilation (kernel/.so/C++): BAKE it, on a SEPARATE LEAF tag, FROM the
             pinned base. Dense serves on :int8, MoE serves on :moe; neither runs the other.
  ```
  ```
    vllm-xpu-base:0.23.0  (digest-pinned, NEVER patched)
      +-- :int8   (+ compiled int8 linear kernel)   <- dense int8 only
      +-- :moe    (+ compiled MoE kernel)            <- MoE only
      +-- pure-Python patches (quark.py ...) -> NOT baked; mounted per-container
  ```
- Forward note (not urgent): the current quark.py patch is a WHOLE-FILE replacement, brittle
  across vLLM bumps. A minimal monkeypatch via `sitecustomize` (already used elsewhere in the
  repo) is the more durable form.

### rdy_to_serve/  (golden path)
```
  rdy_to_serve/
    README.md                  index of models + the START-WITH-v0230 rule
    _common/                   model-AGNOSTIC plumbing ONLY (see contract below)
      lib.sh                   wait_healthy(), gen_probe(), base docker-run flags, bench wrapper
    <model>/                   one dir per served config, SELF-CONTAINED
      serve.sh                 model-specific knobs + `source ../_common/lib.sh`
      patches/                 the patches THIS model mounts (copied in, local)
      README.md               recipe + verified perf + IMAGE DIGEST pinned + sweep manifest line
```

`_common/` contract (this is what makes the shared dir safe):
- ONLY content that is byte-identical for EVERY model AND cannot encode model-specific
  behavior: the health-wait loop, the gen-probe, the gpu-run invocation, the bench wrapper,
  the base `docker run` flags (--device /dev/dri, --ipc=host, shm, cache mounts).
- If a snippet would ever need `if MODEL is MoE` -> it does NOT belong in `_common/`; it
  stays in the model's `serve.sh`.
- Everything risky and model-specific stays LOCAL to the model dir: image digest, TP + the
  #41663 multi-GPU env, graph-capture flags (PIECEWISE/FULL, pass_config, capsizes), which
  patches to mount, VLM text-only, MAXLEN/MAXSEQS. -> a `_common/` change has LOW blast
  radius BY DESIGN.

Note the deliberate tradeoff: `source ../_common/lib.sh` means serve.sh is "self-contained
modulo `_common/`," not literally standalone. That is accepted because `_common/` holds only
low-risk model-agnostic plumbing AND is gated by the sweep rule. The things you most want
frozen per model (patches + model-specific knobs) remain copied into the model dir.


## The SWEEP GATE (the hardline rule, made followable)

Any change to a SHARED/STABLE artifact (`bin/` OR `rdy_to_serve/_common/`) is not done until a
sweep across ALL rdy_to_serve models is green. Two tiers so the rule is actually followed:

```
  SMOKE sweep  (MANDATORY on every bin/ or _common/ change)
    per model: gpu-run -> serve.sh -> wait /health -> 1 gen token -> serve.sh stop
    fast, catches "it no longer boots / no longer generates."

  BENCH sweep  (REQUIRED only when the change could affect PERF)
    per model: the full concurrency sweep (35_sweep_bench). Slow; serialized through gpu-run.
```

- Serialized: one B70, gpu-run lease -> the sweep runs models one at a time (CLAUDE.md).
- Harness: `bin/serve-sweep [--smoke|--bench]` iterates rdy_to_serve/<m>/serve.sh.
- Manifest: each model README carries a line `verified: _common@<sha> bin@<sha> smoke=green YYYY-MM-DD`
  so drift is visible at a glance.


## Single source of truth per recipe (kill the 4x drift)

The 35B-W8A8 recipe currently lives in FOUR places: docs/SERVING.md, scripts/76,
rdy_to_serve/<m>/, and daily_driver_serve.sh. Collapse to one:
- `rdy_to_serve/<m>/serve.sh` = THE executable recipe (source of truth).
- `docs/SERVING.md` -> becomes an INDEX: "serve X -> `cd rdy_to_serve/X && ./serve.sh`",
  plus cross-cutting knowledge (gpu-run, the #41663 env explained, DP vs TP vs PP tradeoffs).
  Stops embedding full per-model commands.
- `daily_driver_serve.sh` -> CALLS the golden path (picks an rdy_to_serve model + wraps it in
  the DP-2-replica + nginx + Open WebUI lifecycle) instead of re-encoding the serve env.
- the `NN_` script that birthed the recipe = frozen history; SERVING points at rdy_to_serve, not NN.

Patches: ONE canonical copy. The blessed copy lives in `rdy_to_serve/<m>/patches/`. `contrib/`
is where patches are DEVELOPED; on graduation, copy into the model dir and mark contrib's as
"dev copy, blessed copy in rdy_to_serve/<m>". Drop the host-side third copy. Optional
pre-commit `sha1sum` check flags when copies diverge.


## Standing rules to graft into CLAUDE.md (once this is adopted)

- scripts/NN_*.sh is APPEND-ONLY. Never edit a committed NN to test a new idea -- copy to the
  next number. Re-run-constantly scripts belong in bin/, not the NN namespace.
- Docker image tags are IMMUTABLE. Never overwrite a tag (no `docker commit` onto an existing
  tag). New build -> new dated/sha tag + a Dockerfile in images/<tag>/. The base image is
  never patched. rdy_to_serve pins the image DIGEST, not the tag.
- A behavior-changing patch is either mounted per-container (pure-Python) or baked on a
  separate leaf tag (compiled) -- never baked into the shared base.
- Any change to bin/ or rdy_to_serve/_common/ requires `bin/serve-sweep --smoke` green across
  all models (and --bench if perf could move) before it is considered done.
- One recipe, one home: rdy_to_serve/<m>/serve.sh. SERVING.md indexes; daily_driver calls in.


## Migration checklist (NOT executed -- do in order, commit each step)

1. [ ] Create `bin/`; move `gpu-run`, `30_serve_w4a8_graph.sh`, `35_sweep_bench.sh`,
       `64_dataparallel_2rep.sh` there. Update the host sync (flat layout) + all callers
       (daily_driver, SERVING, rdy_to_serve). Verify daily_driver still starts.
2. [ ] Create `rdy_to_serve/_common/lib.sh`; lift the model-agnostic plumbing out of the one
       existing serve.sh; re-point it via `source`. Smoke-sweep.
3. [ ] Write `bin/serve-sweep` (--smoke / --bench). Run it; record manifest lines.
4. [ ] Create `images/`; convert scripts/47 (and the other docker-commit builds) into
       Dockerfiles FROM a digest-pinned base; rebuild to dated tags; record digests.
5. [ ] Re-point each rdy_to_serve README/serve.sh to pin image DIGEST (not tag).
6. [ ] Dedupe quark.py: blessed copy in rdy_to_serve/<m>/patches/; mark contrib's as dev;
       drop the host third copy. Add the sha1sum drift check.
7. [ ] Rewrite SERVING.md as an index; make daily_driver call the golden path.
8. [ ] Graft the standing rules into CLAUDE.md.


## Open questions
- Image base: is `vllm-xpu-env:v0230` itself built from a Dockerfile we control, or pulled?
  If pulled, pin it by digest and record the source. If built, it needs an images/ recipe too.
- bin/ vs host flat layout: the host runs scripts FLAT at /mnt/vm_8tb/b70/. Decide whether the
  sync flattens bin/ to that root, or whether the host adopts bin/ too.
