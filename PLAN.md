# PLAN — open design decisions

Captured 2026-05-20 from a strategic discussion that surfaced three
decisions about mlprobe's direction. The current code is at a clean
landing point; these decisions shape what gets built next. Read this
when fresh, then decide.

## What triggered this

I tried to use mlprobe on a real model and hit friction:

- **Dependency wall** — `train_callable: "my_co.training:train"` requires the
  whole work codebase + its deps to be importable in the same Python env
  as mlprobe. For a complex production model that's often impractical.
- **Two-env reality** — many ML projects have a local dev venv (Mac ARM,
  IDE-only deps) AND a Docker training image (Linux + CUDA + full deps).
  mlprobe's docs assume one env; reality has two.
- **MLflow is the system of record** — at MLflow shops, every training
  run already has structured params, metrics, tags, and artifacts logged.
  The "framework-agnostic" framing makes mlprobe ignore all of that and
  re-derive it from code grep, which is doing more work for less value.

In parallel, a separate realization: mlprobe's primitives (`measure()`,
`audit()`, capability detection) are useful *during* a real training run,
not just for external probes that predict before commitment. This was
never the framing; it's a meaningful expansion.

## Insights worth keeping

1. **The dependency problem doesn't have one fix.** Three modes coexist:
   - Audit-only from any env (works today; doesn't need to import user code)
   - Probes inside the training env (install mlprobe into your training venv/container)
   - Probes orchestrated externally via Docker (issue [#1](https://github.com/dnewcome/mlprobe/issues/1), longer-term)

2. **The agnostic protocol cost almost nothing to build but pays nothing
   without users who actually need it.** The user (you) is at an MLflow
   shop. The "shops that need mlprobe but aren't on MLflow" user is
   hypothetical.

3. **Library mode sidesteps the dependency problem entirely.** If mlprobe
   is `import`ed inside the actual training script, you're already in the
   right env. No coordination needed.

4. **Logging probe predictions + library-mode measurements back to MLflow
   creates a predicted-vs-actual feedback loop.** This is the killer
   integration — mlprobe's accuracy becomes measurable and queryable in
   the MLflow UI. That's how a tool stops being hand-wavey and starts
   being trusted.

5. **Hyperparameter sweeps are not the natural mental model; scenarios
   are.** Real-world contact (2026-05-21) revealed that users don't
   think in terms of "declare a sweep, get a Pareto frontier." They
   think in specific questions about their model: "does this scale
   linearly?", "where does parallelization plateau?", "what's the
   cheapest instance that finishes in under an hour?". Each question
   wants a different probe plan and a different analyzer — not a generic
   Cartesian sweep. This is heuristic, not analytical, and often needs
   iteration (run one scenario, look at the answer, decide what to run
   next). See Decision 4.

## Decision 1: framing — MLflow-first vs. agnostic vs. MLflow-only

### Options

- **A. Stay agnostic** (status quo). Three callables, no opinion on
  experiment tracker. Pros: maximally portable; AG News demo stays
  unchanged; sklearn/spaCy users covered. Cons: doing more heuristic
  work to detect what MLflow already structures; the value-add for
  MLflow users is muted.

- **B. MLflow-first, agnostic kept as fallback (RECOMMENDED).** README +
  docs + demos lead with MLflow. The three-callable protocol stays as a
  lower-level interface for non-MLflow users (similar to how Lightning
  has `LightningModule` but you can drop to raw PyTorch). Pros: matches
  your actual use case; doesn't throw away working code; preserves
  optionality. Cons: project identity is slightly diffused.

- **C. MLflow-only.** Drop the agnostic protocol; rip out the AG News
  demo or make it use mlflow even though it's silly to. Pros: sharpest
  identity; smallest surface; deepest possible integration. Cons:
  irreversible; cuts off whatever non-MLflow users might exist; loses
  the protocol as a fallback for weird cases.

### Recommendation

**B.** MLflow-first, protocol as fallback. The protocol is small and
already built — keeping it costs ~0 maintenance and preserves the option
to support W&B / Neptune / others later if you ever care. MLflow becomes
the documented happy path, gets all the deep integration, and the AG
News demo stays as the "manual mode" reference.

### What changes if you go B

- README headline: from "framework-agnostic profiler" to **"production-
  readiness audit, cost prediction, and bottleneck profiling for
  MLflow-tracked training. Works with any framework via a three-callable
  protocol."**
- Documented happy path becomes: `from_mlflow_run(run_id) → ModelSpec →
  audit / profile → log_results_to_mlflow(report)`.
- The generate-modelspec skill grows an MLflow run input mode as the
  primary path; file-based input becomes the fallback.
- Capability auto-detection becomes MLflow-aware first (mlflow.pytorch
  flavor → look at HF Trainer signatures; mlflow.sklearn → cpu_bound).
  Code-grepping becomes the secondary path.

## Decision 2: library mode vs. probe-only mode

### Options

- **A. Probe-only (status quo).** mlprobe is an external tool that
  predicts before training runs. Pros: clear product identity; clean
  abstractions. Cons: hits the dependency wall; doesn't help during real
  training runs you're already paying for; users want both.

- **B. Probe mode + library mode (RECOMMENDED).** Promote `measure()`,
  `audit()`, and a new `mlprobe.profile()` context manager to public API.
  Users can either run probes ahead of time (existing mode) OR
  instrument their actual training (new mode). Pros: sidesteps the dep
  problem; library mode is what most people will reach for first; same
  primitives serve both. Cons: docs surface roughly doubles; need to
  explain when to use which.

- **C. Library-only.** Drop the probe orchestration; mlprobe becomes
  pure instrumentation. Pros: massively simpler. Cons: throws away the
  scaling extrapolation + Pareto frontier work, which is the original
  value prop.

### Recommendation

**B.** Add library mode as a second front door without removing the
first. The two modes share ~80% of the code (measure, audit, report
formatting) — the new surface is mostly a single
`with mlprobe.profile(spec=...) as p: ...` context manager and a
single-run report shape that skips the scaling fits.

### What changes if you go B

- Two new issues to file:
  1. Promote `measure()` and `audit()` to public API + library-mode
     docs (small, mostly documentation, days)
  2. Add `mlprobe.profile()` top-level context manager for single-run
     reports (real feature work, ~1 week)
- README adds a second-quickstart section showing library-mode usage
  inside a training script.
- Stage profiling ([#13](https://github.com/dnewcome/mlprobe/issues/13))
  becomes much more valuable — in library mode it measures the actual
  production run's bottlenecks, not extrapolations.

## Decision 3: how deep to go on MLflow integration

### Options

- **A. Thin integration** (status quo of issue [#17](https://github.com/dnewcome/mlprobe/issues/17)).
  `from_mlflow_run(run_id) -> ModelSpec` and
  `log_results_to_mlflow(report)` as two helper functions. Pros: small,
  contained, optional dep. Cons: doesn't fully exploit what MLflow
  provides.

- **B. Native integration (RECOMMENDED).** Multiple granular logging
  primitives:
  - `log_predictions()` — predicted cost/time/memory/quality as metrics
  - `log_audit_findings()` — finding codes as tags, full audit as artifact
  - `log_stage_profile()` — per-stage timings as metrics (library mode)
  - `with mlprobe.profile(mlflow_run=...)` — context manager that does all
    of it automatically inside a training script
  - Pre-flight audit gate that can `raise` and fail the MLflow run with
    a clear blocker reason visible in the UI
  - Predicted-vs-actual comparison built in (when both predicted and
    actual cost end up on the same run, expose the delta as a tag)

  Pros: mlprobe becomes native to the MLflow UI; every value the tool
  produces is visible where users already live; predicted-vs-actual
  feedback loop makes the tool self-validating. Cons: more surface to
  maintain; more places MLflow changes can break us.

- **C. MLflow under the hood, no user-visible MLflow APIs.** Users still
  write specs; mlprobe transparently logs to MLflow if it's configured.
  Pros: zero user effort. Cons: surprising magic; hard to debug; the
  user is best served seeing the integration explicitly.

### Recommendation

**B.** The predicted-vs-actual feedback loop alone is worth the surface
area. The other primitives compose nicely with library mode (decision 2)
— `mlprobe.profile(mlflow_run=...)` is the unified entry point that uses
all of them under the hood, so users don't have to call each one
themselves.

### What changes if you go B

- Issue [#17](https://github.com/dnewcome/mlprobe/issues/17) grows
  significantly — should be re-scoped or replaced with a richer issue
  (or split into 3-4: read-side, write-side, library-mode integration,
  pre-flight gate).
- New issue specifically for the **predicted-vs-actual feedback loop**
  as a measurable design goal — it's the killer differentiator and
  deserves explicit tracking.
- `mlflow` as a `[mlflow]` extra in pyproject (already proposed in #17).
- Optional but valuable: the AG News demo gets an MLflow-tracked variant
  to demonstrate the integration end-to-end.

## Decision 4: scenarios vs. sweeps as the primary user surface

### Options

- **A. Sweep-and-Pareto (status quo).** Users declare hyperparameter
  sweeps (NumericSweep / CategoricalSweep) and target lists; mlprobe runs
  the Cartesian product and reports the Pareto frontier. Pros: clean,
  fully built. Cons: doesn't match how engineers actually think — they
  have questions ("does this scale linearly?", "where does
  parallelization plateau?") and have to translate them into sweep
  configurations.

- **B. Scenarios as first-class objects (RECOMMENDED).** Promote
  question-shaped probe patterns to the primary user surface. Each
  scenario knows the question it answers, the probe plan needed,
  and a domain-specific analyzer. Sweeps become one scenario
  (``scaling_with_n``) among many.

  ```python
  from mlprobe.scenarios import scaling_with_n, parallelization_effect, cheapest_instance

  scaling_with_n.run(spec, target="g5.xlarge")
  # → "Scales as O(N^1.04). R²=0.99 on 5 subset sizes. Likely linear."

  parallelization_effect.run(spec, target="c5.4xlarge", n_cpus=[1,2,4,8,16])
  # → "Speedup 1.0/1.8/3.1/4.5/5.2x. Sub-linear past 4 CPUs (Amdahl ~ 0.85)."

  cheapest_instance.run(spec, max_wall_clock_s=3600, candidates=[...])
  # → "g5.xlarge meets budget at $1.01 (~42min). c5.4xlarge doesn't (9.2hr).
  #    p3.2xlarge $4.50 (~24min) — only worth it if you need <30min."
  ```

  Pros: matches the actual mental model engineers use; each scenario can
  have a focused analyzer (not just curve fitting); cheap individually
  so users can iterate. Cons: more abstractions to maintain; the
  existing Pareto frontier becomes one output shape among many, not THE
  output shape.

- **C. Both layers — sweeps stay primary, scenarios as a thin sugar
  layer on top.** Pros: lowest risk; nothing existing changes. Cons:
  doesn't actually address the insight that sweeps aren't the right
  mental model; just papers over it.

### Recommendation

**B.** The existing sweep machinery is exactly the implementation of one
specific scenario (``scaling_with_n``) — Cartesian over subset fractions
+ linear/loglinear/power fit. Other scenarios reuse the same probe
infrastructure (measurement, subprocess launcher, runtime resolver) but
plan their probes and analyze results differently.

### Catalog of scenarios worth filing as separate issues

| Scenario | Question it answers |
|---|---|
| ``scaling_with_n`` | Does this scale linearly with data size? What's the exponent? |
| ``parallelization_effect`` | Where does adding CPUs/GPUs plateau? |
| ``gpu_vs_cpu`` | Is GPU worth it for this specific model? |
| ``cheapest_instance`` | Lowest-cost target that meets a wall-clock budget |
| ``vram_headroom`` | Largest batch size that fits in target VRAM |
| ``mixed_precision_payoff`` | What does fp16/bf16 buy us in time / cost / quality? |
| ``stage_bottleneck`` | Which stage dominates wall-clock? CPU- or GPU-bound? (becomes the user-facing surface for [#13](https://github.com/dnewcome/mlprobe/issues/13)) |
| ``regression_guard`` | Did this perf optimization keep quality? (the user-facing surface for [#14](https://github.com/dnewcome/mlprobe/issues/14)) |
| ``checkpoint_robustness`` | Does the checkpointing actually work mid-training? (the user-facing surface for [#9](https://github.com/dnewcome/mlprobe/issues/9) / [#11](https://github.com/dnewcome/mlprobe/issues/11)) |
| ``incremental_amortization`` | How much cheaper is warm-start vs from-scratch? (the user-facing surface for [#10](https://github.com/dnewcome/mlprobe/issues/10)) |
| ``spot_savings`` | What does spot pricing save with my checkpointing setup? (the user-facing surface for [#12](https://github.com/dnewcome/mlprobe/issues/12)) |
| ``deployment_readiness`` | Combined: model size + inference VRAM + cold-load time (combines [#8](https://github.com/dnewcome/mlprobe/issues/8) + [#15](https://github.com/dnewcome/mlprobe/issues/15)) |

### What changes if you go B

- New ``mlprobe/scenarios/`` package with a ``Scenario`` protocol
  defining ``run(...)`` and a ``Result`` shape.
- Existing sweep + Pareto code stays — it becomes the implementation
  of ``scaling_with_n`` and a generic ``hyperparameter_sweep`` scenario
  for users who really do want Cartesian exploration.
- README leads with scenario examples, not sweep examples.
- The generate-modelspec skill output changes — instead of
  "configure your sweep here", it suggests "you'll probably want to run
  these scenarios first: [list]".
- Most existing GH issues get a "implements scenario X" tag, since the
  scenarios catalog above reorganizes the backlog around user-facing
  questions instead of internal capabilities.
- ``ProbeConfig`` becomes more of an internal detail; users mostly
  interact with scenario parameters instead.

## Cross-cutting: how does this resolve the dependency story

If you go B/B/B above:

| Scenario | mlprobe workflow |
|---|---|
| Local Mac dev only, no training here | Install mlprobe in dev venv; write spec by hand or via skill; run `audit()` only |
| Single venv with model deps + mlprobe | `pip install mlprobe[mlflow]`; do everything (probes, library mode, MLflow logging) |
| Docker training image | Add `mlprobe[mlflow]` to image deps; use library mode inside the container; predictions log to the same MLflow run |
| Two-env (Mac dev + Docker training) | Dev venv: audit-only against MLflow runs. Container: library mode + MLflow logging. Best of both. |

The dependency wall disappears for most users because mlprobe gets
installed alongside the existing training env (whichever env that is),
not into its own isolated venv.

## What's already filed vs. what would need filing

### Already filed (relevant to these decisions)
- [#1](https://github.com/dnewcome/mlprobe/issues/1) Docker launcher — still relevant, less central if library mode wins
- [#7](https://github.com/dnewcome/mlprobe/issues/7) YAML spec loading — orthogonal, still useful
- [#13](https://github.com/dnewcome/mlprobe/issues/13) Stage-level profiling — *becomes the highest-value issue* in library mode
- [#15](https://github.com/dnewcome/mlprobe/issues/15) Model size — orthogonal, still useful
- [#16](https://github.com/dnewcome/mlprobe/issues/16) sagebaker ↔ mlprobe relationship — needs minor update if mlprobe goes MLflow-first
- [#17](https://github.com/dnewcome/mlprobe/issues/17) MLflow integration — needs to be re-scoped or split if decision 3 lands on B

### Would need filing if B/B/B/B
- Library mode v1: promote `measure()`/`audit()` to public API + docs
- Library mode v2: `mlprobe.profile()` top-level context manager
- Pre-flight audit gate (capability to `raise` on incompatibilities)
- Predicted-vs-actual feedback loop as an explicit feature
- README rewrite for MLflow-first + scenarios-first framing
- AG News demo MLflow variant
- **Scenario framework + baseline scenarios** — meta-issue defining the
  `Scenario` protocol and an issue per scenario in the catalog above.
  Most of the existing scenario-shaped issues (#9, #10, #11, #12, #13,
  #14) get re-tagged as "implements scenario X" but their content stays
  largely intact.

### Would need updating
- README (headline + framing)
- generate-modelspec skill (MLflow input as primary)
- Issue #17 (re-scope to richer integration)

## Recommended path if you don't want to think harder

Commit to **B, B, B, B** (MLflow-first with protocol fallback; both
modes; native MLflow integration; scenarios as primary user surface).
The pieces compose: scenarios are the question-shaped front end,
library mode is how scenarios get invoked during real training runs,
MLflow is where scenario results land, and the agnostic protocol stays
as an escape valve. Re-scope #17, file the four library-mode/MLflow
issues + the scenario catalog issues, then implement in this order:

1. **README + framing change** (afternoon — docs only, sets scenarios
   + MLflow as the primary mental model)
2. **Promote `measure()` and `audit()` to public API** (small,
   immediate user value: people can start instrumenting today)
3. **Define the `Scenario` protocol + ship 2–3 baseline scenarios**
   (`scaling_with_n` as the obvious one since it's just a refactor of
   existing code; plus `cheapest_instance` and `parallelization_effect`
   as the next two highest-value). This proves the abstraction holds
   before investing more.
4. **`from_mlflow_run()` helper** (cuts skill complexity, removes the
   biggest manual-spec friction)
5. **`mlprobe.profile()` context + MLflow logging** (the unified entry
   point users will actually use; scenarios run inside it)
6. **Stage profiling [#13](https://github.com/dnewcome/mlprobe/issues/13)** —
   becomes the measurement primitive for the `stage_bottleneck` scenario
7. **More scenarios as needed** based on real-world use — `gpu_vs_cpu`,
   `vram_headroom`, `mixed_precision_payoff`, etc.
8. **Predicted-vs-actual** once both predictions and library-mode
   measurements are landing on MLflow runs

That sequence builds incremental value, validates the scenario
abstraction before doubling down, and ends with the tool that matches
the discussion's eventual shape.

## Things I'd push back on if you proposed them

- **Going MLflow-only (C on decision 1)** — irreversible, throws away
  the protocol which costs nothing to keep. Even if you only ever use
  MLflow yourself, leaving the agnostic path means contributors with
  W&B / Neptune / homegrown trackers can extend later.
- **Library-only (C on decision 2)** — the scaling extrapolation + Pareto
  frontier IS the original value prop. Probe mode is what makes mlprobe a
  planning tool, not just a measurement library.
- **Magic MLflow (C on decision 3)** — implicit cross-cutting integrations
  are hard to debug when they break. Users are best served by explicit
  primitives even if a top-level convenience wrapper exists.
- **Scenarios on top of sweeps (C on decision 4)** — adding scenarios as
  thin sugar over the existing sweep API doesn't address the actual
  insight (that engineers think in questions, not parameter grids). It
  papers over the wrong abstraction instead of replacing it. Better to
  promote scenarios to first-class and let sweeps live as one specific
  scenario implementation.

## Things this plan does *not* decide

- Whether to support W&B / Neptune / other trackers. Not now; could be
  added behind a tracker-abstraction later if you ever want.
- Whether to publish to PyPI. Not now; install from git is fine for an
  early project.
- Whether the project needs a separate website / hosted docs. Not now;
  README + per-issue specs are enough.
- What to do about sagebaker integration beyond
  [#16](https://github.com/dnewcome/mlprobe/issues/16). The
  `from_sagebaker_plugin()` adapter remains optional.
- The exact catalog of scenarios. The table in Decision 4 is a
  starter set, not exhaustive. New scenarios will come up from real
  use; the framework should make adding them cheap.
- Whether scenarios should be composable (e.g. a `diagnose` scenario
  that runs `scaling_with_n` + `stage_bottleneck` + `gpu_vs_cpu` and
  produces a combined recommendation). Probably yes eventually, but
  not in the first cut.

---

# Inference profiling (added 2026-05-29) — issues #8 + #15

Everything above is about *training*. mlprobe has nothing on the
**inference / deployment** side yet, and this section plans it. It was
written after the launcher work (#1/#24), so it assumes the pieces shipped
this cycle exist: scenarios, library mode (`profile()`), eval-only
(`evaluate_existing`), the `ProbeLauncher` protocol + Docker launcher, and
dataset profiling (incl. `token_length_profile`). Issues
[#8](https://github.com/dnewcome/mlprobe/issues/8) (inference profiling) and
[#15](https://github.com/dnewcome/mlprobe/issues/15) (model size) predate all
of that and reference the old `ml_baker/` names — this section supersedes
their design notes.

If you don't want to read the whole thing: jump to
**"Recommended path if you don't want to think harder"** at the end. It's a
concrete default you can greenlight as-is.

## What inference profiling is

Profile a *trained* model for **deployment**, which is a different question
from training:

- Inference VRAM ≪ training VRAM (no optimizer state, no gradients, often
  quantized).
- Latency is batch-size-sensitive; cold-load vs warm-request matters for
  serverless/autoscaling.
- Cost is per-request (cost-per-1k-requests), not per-training-run.
- The serving hardware is usually *different* from training (train on a
  `p4d`, serve on a `g4dn` or CPU).

It is a clean fit for mlprobe's scope boundary (see
[[mlprof-scope-boundary]] in memory): pure **measurement** (load time,
latency, VRAM, throughput, artifact size) plus **surfacing** (cost-per-1k, a
deployment Pareto, audit findings against *your* declared SLA). It never picks
the instance for you — it measures and you decide.

## How it fits the architecture we now have

The original issue framed inference as an after-training runner pass that
appends an "INFERENCE" section to the training report. Given what shipped this
cycle, three changes make it cleaner and more powerful:

1. **Mirror eval-only.** The core primitive is a standalone
   `profile_inference(spec, artifact_path, target, batch_sizes=...)` that
   profiles an *existing* artifact, decoupled from any training run — exactly
   analogous to `evaluate_existing`. "Run after training" becomes a thin
   convenience on top, not the core. This lets you profile a model you already
   have, on a serving box, without re-running anything.

2. **Reuse launchers via a mode flag.** Add a discriminator to the probe input
   JSON (`mode: "train_eval" | "inference"`) and dispatch inside
   `python -m mlprobe.probe`. Then *every* launcher (subprocess, Docker, later
   SageMaker) runs inference probes with **zero launcher changes** — and you
   can profile on the real *serving* container, which is often different from
   the training image. (This is a payoff of the launcher refactor.)

3. **Scenarios are the surface.** Consistent with Decision 4, the user-facing
   surface is scenario(s) — `latency_vs_batch_size`,
   `cheapest_inference_target`, `deployment_readiness` — with the inference
   probe + report underneath. Inference metrics also flow into
   `baseline_compare`'s metric vector for free (it already compares latency).

## The distinctive piece: training → inference linkage

The most mlprobe-ish part, and the thing nothing else gives you: **a lot of
inference performance is decided at training time, and most of those decisions
are already visible in the `ModelSpec`.** So mlprobe can connect the two sides.

Training decisions that determine inference performance:

- **Architecture / size** (dominant): encoder/base-model, depth, width,
  hidden size, **vocab/embedding size** → param count, artifact size, VRAM,
  FLOPs-per-request, latency. These are *structural hyperparameters mlprobe
  already sweeps*, so the inference cost of "distilbert vs bert" is a
  training-choice axis.
- **Precision**: training in bf16/fp16 sets the saved weights' dtype; serving
  on a GPU without bf16 (T4/V100) forces an fp32 fallback → ~2× VRAM +
  different latency. mlprobe can predict this *from the spec* (it knows
  training precision + the inference target's GPU). QAT vs post-training quant
  also reshapes the whole latency/quality curve.
- **What `train()` saves**: full checkpoint / optimizer state vs inference-only
  weights → bloated artifact + slow cold load (serverless killer). Save
  *format* (safetensors/ONNX/TorchScript vs pickle) → load time, sometimes
  latency. **LoRA**: serving the adapter unmerged → base-load + merge at cold
  start → slower load + small per-request overhead.
- **Sequence length** (LLM/NLP): `max_seq_len` / `max_position_embeddings`
  caps inference context and sets KV-cache size + quadratic attention cost.
  Connects directly to `token_length_profile` — if p99 tokens = 512 but you
  trained/serve at 2048, you pay for headroom you never use.
- **Correctness gotchas** (quality, not speed): BatchNorm running stats depend
  on training batch size; inference precision differing from eval precision
  can drift quality → "re-validate."

This motivates two features beyond raw measurement, both surface-not-prescribe:

- **Provenance tagging.** Each inference profile is tagged with the training
  config that produced the artifact, so the deployment Pareto's "architecture"
  axis *is* the training choice (distilbert → 250MB / 12ms / $0.40 vs bert →
  440MB / 21ms / $0.70). The inference cost of a training decision becomes
  legible.
- **A train→infer mismatch audit** (cheap, pre-probe, like the existing
  capability `audit`), surfacing findings such as:
  - trained bf16 → serving on T4 (no bf16) → expect fp32 fallback, ~2× VRAM
  - `max_seq_len=2048` but `token_length_profile` p99=512 → unused
    KV-cache/attention cost
  - artifact includes optimizer state → strip for deployment (size/cold-load)
  - LoRA adapter unmerged → slower cold load

The mismatch audit is the highest-value, most-distinctive piece — it fuses the
dataset profiler, the training spec, and the inference target into findings
nothing else produces. **Implication:** the standalone primitive should still
*accept/propagate training provenance* (config + precision + the spec) so the
linkage works whether you profile right after training or against an artifact
you already have.

## Protocol + spec additions

```python
class InferFn(Protocol):
    def __call__(self, artifact_path, inputs, runtime,
                 batch_size, n_warmup) -> InferResult: ...

@dataclass
class InferResult:
    batch_latencies_ms: list[float]   # one per measured batch, post-warmup
    load_time_s: float                # cold load: disk -> ready
    error: str | None = None
```

(Tightened vs the issue: per-**batch** times, not per-request — throughput =
`batch_size / batch_time`, per-request = `batch_time / batch_size`. Removes the
ambiguity.)

`ModelSpec` gains: optional `inference_callable`, `inference_targets` (a
*separate* list from training `targets` — agreed), a `max_latency_ms` declared
constraint (for the gate), and an `InferenceProbeConfig(batch_sizes, n_warmup,
n_measure, timeout_seconds)`.

## Metrics → report → Pareto

- Latency-vs-batch-size fit (reuse `mlprobe/scaling.py`).
- Derived **throughput** (`batch_size / batch_latency_s`) and
  **cost-per-1k-requests** (`$/hr × 1000 / throughput`).
- `InferenceReport` (its own shape, like `ProfileReport`): load_time,
  p50/p95/p99 latency at the configured batch size, throughput, inference
  VRAM, cost-per-1k, artifact size.
- Audit findings: `inference_vram_overshoot`, `inference_latency_overshoot`
  (vs declared `max_latency_ms`), `inference_cold_load_slow`, plus the
  train→infer mismatch findings above.
- A **deployment Pareto** over `(cost_per_1k, quality)` across
  (architecture × inference_target × batch_size) — same data, different axes
  from the training Pareto.

## Model size (#15) — the cheap prerequisite

Independent, ~an hour, feeds everything above. After a successful training
probe, `du` the artifact dir → `artifact_size_mb` (and
`largest_checkpoint_size_mb`) on `ProbeResult`; surface on `GroupSummary` +
report line; optional user `params_callable` for param count (keeps mlprobe
framework-agnostic); `large_artifact` audit finding (threshold off by default).
Answers "how many replicas fit per GPU?" alongside inference VRAM.

## LLM / generation latency

Issue #8 lists streaming as out of scope, but since the LoRA/LLM work landed,
**generation latency is probably the metric that actually matters there**:
time-to-first-token (TTFT), inter-token latency, tokens/sec — not batch
throughput. Batch latency is right for classifiers/embedders/rerankers; TTFT is
right for generation. Decision below on whether to include it now.

## Phasing (each independently shippable)

0. **#15 model size** — tiny standalone warmup.
1. **Inference protocol + probe primitive + `profile_inference()`** — eval-only
   style; synthetic infer callable + `in_process` tests (no ML deps);
   `InferenceReport`; the probe-mode dispatch in `mlprobe.probe`.
2. **Fits + cost-per-1k + audit findings (incl. train→infer mismatch) +
   deployment Pareto.**
3. **Scenarios** — `latency_vs_batch_size`, `cheapest_inference_target`,
   `deployment_readiness` (combines #8 + #15 + inference VRAM; the catalog
   entry from Decision 4).
4. *(optional)* runner auto-runs inference probes after a training sweep on the
   highest-quality artifact per architecture.

## Open decisions (with recommendations)

- **Entry point** — standalone `profile_inference(artifact)` primitive
  (RECOMMENDED, mirrors eval-only) vs. after-training runner pass only. Best:
  primitive first, runner integration as phase 4. *The primitive must still
  accept training provenance so the mismatch audit works.*
- **Surface** — scenarios-first (RECOMMENDED, matches Decision 4) vs.
  report-section-first.
- **LLM generation latency (TTFT/tokens-sec)** — defer to a follow-up
  (RECOMMENDED for a first cut: batch latency/throughput/VRAM/load first) vs.
  include now. Lean "include soon after" given the LLM direction.
- **Quality during inference** — skip (RECOMMENDED; quality came from
  training-time `evaluate`); the inference probe is operational metrics only,
  with a "re-validate if inference precision ≠ eval precision" finding.

## Out of scope (separate issues / later)

- Quantization (INT8/INT4) + the quality-vs-latency tradeoff.
- Multi-replica throughput projection ("8 replicas behind a load balancer").
- SLA enforcement / autoscaling cost modeling.
- On-the-fly size estimation from layer specs (just measure the artifact).

## Recommended path if you don't want to think harder

Commit to **primitive-first + scenarios-first + defer-TTFT**, and build in this
order — each step is small and lands value on its own:

1. **#15 model size** (artifact `du` in the probe + report line + finding).
   Trivial, immediately useful, no new concepts.
2. **`profile_inference(spec, artifact_path, target, batch_sizes=...)`** — the
   eval-only-style primitive, with a synthetic infer callable and `in_process`
   tests. Add `mode: "inference"` dispatch to `mlprobe.probe` so all launchers
   work. Returns an `InferenceReport` (load, p50/p95/p99, throughput, VRAM,
   cost-per-1k, size).
3. **Train→infer mismatch audit + provenance tagging** — the distinctive
   piece; cheap, pure surfacing, fuses dataset profiler + spec + target.
4. **Scenarios** — `latency_vs_batch_size` and `deployment_readiness` as the
   front end.
5. Later: TTFT/generation latency; runner auto-profiling post-training.

That sequence gives a usable "profile any artifact for deployment" tool after
step 2, the unique linkage value after step 3, and the polished surface after
step 4 — without committing to the big stuff (generation latency, multi-replica)
until real use asks for it.

---

# Pipeline & steps (added 2026-05-29) — generalize beyond train→eval

mlprobe's model of a run is essentially **`train()` → `evaluate()`** (with
dataset profiling before and inference planned after). But real pipelines —
especially clustering / dedup / entity-resolution like the brand-clustering
work — have *other steps* that often dominate cost and aren't "training" at
all: preprocess, **embed**, block/index, **cluster**, postprocess. For dedup
the clustering step *is* the model; there's no gradient "training."

Two things are wanted (both selected): **(1)** profile a whole pipeline with a
per-stage breakdown + bottleneck analysis (this is issue
[#13](https://github.com/dnewcome/mlprobe/issues/13)), and **(2)** probe any
single step standalone — embed, block, cluster — like `profile_inference` does
for serving.

## Decision 6: generalize the protocol, or layer steps on the fixed roles?

- **A. Layer (RECOMMENDED).** Keep `train`/`evaluate`/`infer` as named,
  well-supported roles, but add a general *step* concept underneath. A spec may
  declare an explicit `pipeline: list[StepSpec]`; if it doesn't, the default
  pipeline is `[train, evaluate]` — so everything built today is just the
  two-step default. Probe mode runs the declared steps in sequence, measuring
  each; library mode's existing `p.stage()` is the same concept. Adds
  standalone step probing without breaking a single existing spec/example.

- **B. Replace with a general Step/DAG model.** Drop the privileged roles;
  everything is a `Step` with typed inputs/outputs; a pipeline is a DAG. Most
  general; biggest rewrite; breaks existing specs, the scenarios, the examples,
  and the audit's train-centric findings. Not worth it now.

- **C. Library-mode stages only (status quo + #13).** Add bottleneck findings
  to `ProfileReport` but don't touch probe mode. Cheapest, but doesn't give
  standalone step probing (selection #2). Good as *phase 0*, not the endpoint.

**Recommendation: A.** It makes `train→eval` one instance of a pipeline,
unifies with library-mode stages and inference ("infer is just a step"), and
delivers both selections without a rewrite.

## What a "step" is

```python
@dataclass
class StepSpec:
    name: str
    callable: str                 # dotted path to a StepFn
    consumes: str | None = None   # "dataset" | a prior step name | "artifact"
    resource_hint: str | None = None  # "cpu_bound" | "gpu_bound" (for findings)
    params: dict = field(default_factory=dict)

class StepFn(Protocol):
    def __call__(self, inputs, runtime, **params) -> StepResult: ...

@dataclass
class StepResult:
    outputs: Any                  # handed to the next step (or an artifact path)
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
```

mlprobe measures each step externally with the existing psutil/NVML sampler —
the same machinery as train probes. `train`/`evaluate`/`infer` become
well-known step shapes (e.g. a train step `consumes="dataset"` and outputs an
artifact; eval `consumes="artifact"`).

## Pipeline in one run (selection #1, = #13 done properly)

Spec declares `pipeline` (or defaults to train+eval). The probe runs the steps
in order, threading each step's `outputs` into the next's `inputs`, measuring
each. The report gains a per-stage breakdown (wall-clock %, peak RSS/VRAM, GPU
util per step) and a **bottleneck finding**: which step dominates wall-clock
and whether it's CPU- or GPU-bound (e.g. "clustering = 92% of wall-clock,
CPU-bound — GPU on this target is idle for most of the run"). This is exactly
#13, and it works in probe mode, not just library mode. Library mode's
`p.stage()` already does the timing half; this adds per-stage resource sampling
+ the finding, and brings the same to probe mode.

## Standalone step probe (selection #2)

`profile_step(spec, step_name, inputs=..., target=...)` runs one step against
provided or loaded inputs — eval-only style (mirrors `evaluate_existing` /
`profile_inference`). So you can probe just the embedding pass or just
clustering, in isolation, on a chosen target. Because `scaling_with_n` already
sweeps subset sizes, it composes: "how does the clustering step scale with N?"
becomes `scaling_with_n` pointed at one step — which is exactly the
brand-clustering question (the Σ-pairs blowup the `block_size_profile` predicts,
now *measured* per-step).

## Scenarios

- `stage_bottleneck` (the Decision-4 catalog entry) is the surface for
  whole-pipeline profiling.
- `scaling_with_n` / `parallelization_effect` gain an optional `step=` to
  target a single step instead of the whole run.

## How it connects

- **#13** stage profiling — this *is* #13, generalized to a declared pipeline.
- **Library mode** `p.stage()` — same concept; unify so a real run and a probe
  describe stages the same way.
- **Inference (#8)** — `infer` becomes just another step; the pipeline model
  subsumes it.
- **Dataset profiling** — a "preprocess"/"embed" step's measured cost sits next
  to the *predicted* shape cost from `block_size_profile` (predicted vs actual).
- **Scope boundary** — still measure + surface. The bottleneck finding states
  the fact ("clustering dominates, CPU-bound"); it doesn't tell you to rewrite
  the algorithm.

## Phasing (each independently shippable)

0. **Bottleneck analysis + findings on the existing library-mode
   `ProfileReport`** (#13, additive, small) — delivers selection #1 in library
   mode immediately, no protocol change.
1. **`StepSpec`/`StepFn`/`StepResult` + optional `pipeline` on `ModelSpec`;**
   probe binary runs the declared pipeline (default = train+eval) with
   per-step measurement; report per-stage breakdown + bottleneck finding in
   probe mode.
2. **`profile_step()` standalone primitive** (eval-only style) — selection #2.
3. **`stage_bottleneck` scenario; `step=` on `scaling_with_n` /
   `parallelization_effect`.**
4. **Migrate inference (#8) to be "just a step,"** unifying the two plans.

## Out of scope (for now)

- Full DAG / branching pipelines (linear sequence first; most pipelines are).
- Caching step outputs across probes (valuable later — embed once, sweep
  clustering thresholds cheaply; pairs with eval-only's spirit).
- Distributed / multi-node steps.

## Recommended path if you don't want to think harder

Commit to **Decision 6 = A (layer)** and build:

1. **Phase 0** first — bottleneck findings on the library-mode `ProfileReport`.
   It's small, additive, needs no protocol change, and immediately answers
   "which stage dominates?" on any real run you instrument with `p.stage()`.
2. Then **phase 1** (declared `pipeline` + per-step probe measurement), which
   is the real generalization and unblocks everything else.
3. Then **phase 2** (`profile_step`) for standalone step probing.

Phases 0–2 give a pipeline-aware mlprobe that profiles whole runs *and*
individual steps, with `train→eval` still working untouched as the default.
