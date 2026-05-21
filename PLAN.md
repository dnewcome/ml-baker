# PLAN — open design decisions

Captured 2026-05-20 from a strategic discussion that surfaced three
decisions about mlprof's direction. The current code is at a clean
landing point; these decisions shape what gets built next. Read this
when fresh, then decide.

## What triggered this

I tried to use mlprof on a real model and hit friction:

- **Dependency wall** — `train_callable: "my_co.training:train"` requires the
  whole work codebase + its deps to be importable in the same Python env
  as mlprof. For a complex production model that's often impractical.
- **Two-env reality** — many ML projects have a local dev venv (Mac ARM,
  IDE-only deps) AND a Docker training image (Linux + CUDA + full deps).
  mlprof's docs assume one env; reality has two.
- **MLflow is the system of record** — at MLflow shops, every training
  run already has structured params, metrics, tags, and artifacts logged.
  The "framework-agnostic" framing makes mlprof ignore all of that and
  re-derive it from code grep, which is doing more work for less value.

In parallel, a separate realization: mlprof's primitives (`measure()`,
`audit()`, capability detection) are useful *during* a real training run,
not just for external probes that predict before commitment. This was
never the framing; it's a meaningful expansion.

## Insights worth keeping

1. **The dependency problem doesn't have one fix.** Three modes coexist:
   - Audit-only from any env (works today; doesn't need to import user code)
   - Probes inside the training env (install mlprof into your training venv/container)
   - Probes orchestrated externally via Docker (issue [#1](https://github.com/dnewcome/mlprof/issues/1), longer-term)

2. **The agnostic protocol cost almost nothing to build but pays nothing
   without users who actually need it.** The user (you) is at an MLflow
   shop. The "shops that need mlprof but aren't on MLflow" user is
   hypothetical.

3. **Library mode sidesteps the dependency problem entirely.** If mlprof
   is `import`ed inside the actual training script, you're already in the
   right env. No coordination needed.

4. **Logging probe predictions + library-mode measurements back to MLflow
   creates a predicted-vs-actual feedback loop.** This is the killer
   integration — mlprof's accuracy becomes measurable and queryable in
   the MLflow UI. That's how a tool stops being hand-wavey and starts
   being trusted.

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

- **A. Probe-only (status quo).** mlprof is an external tool that
  predicts before training runs. Pros: clear product identity; clean
  abstractions. Cons: hits the dependency wall; doesn't help during real
  training runs you're already paying for; users want both.

- **B. Probe mode + library mode (RECOMMENDED).** Promote `measure()`,
  `audit()`, and a new `mlprof.profile()` context manager to public API.
  Users can either run probes ahead of time (existing mode) OR
  instrument their actual training (new mode). Pros: sidesteps the dep
  problem; library mode is what most people will reach for first; same
  primitives serve both. Cons: docs surface roughly doubles; need to
  explain when to use which.

- **C. Library-only.** Drop the probe orchestration; mlprof becomes
  pure instrumentation. Pros: massively simpler. Cons: throws away the
  scaling extrapolation + Pareto frontier work, which is the original
  value prop.

### Recommendation

**B.** Add library mode as a second front door without removing the
first. The two modes share ~80% of the code (measure, audit, report
formatting) — the new surface is mostly a single
`with mlprof.profile(spec=...) as p: ...` context manager and a
single-run report shape that skips the scaling fits.

### What changes if you go B

- Two new issues to file:
  1. Promote `measure()` and `audit()` to public API + library-mode
     docs (small, mostly documentation, days)
  2. Add `mlprof.profile()` top-level context manager for single-run
     reports (real feature work, ~1 week)
- README adds a second-quickstart section showing library-mode usage
  inside a training script.
- Stage profiling ([#13](https://github.com/dnewcome/mlprof/issues/13))
  becomes much more valuable — in library mode it measures the actual
  production run's bottlenecks, not extrapolations.

## Decision 3: how deep to go on MLflow integration

### Options

- **A. Thin integration** (status quo of issue [#17](https://github.com/dnewcome/mlprof/issues/17)).
  `from_mlflow_run(run_id) -> ModelSpec` and
  `log_results_to_mlflow(report)` as two helper functions. Pros: small,
  contained, optional dep. Cons: doesn't fully exploit what MLflow
  provides.

- **B. Native integration (RECOMMENDED).** Multiple granular logging
  primitives:
  - `log_predictions()` — predicted cost/time/memory/quality as metrics
  - `log_audit_findings()` — finding codes as tags, full audit as artifact
  - `log_stage_profile()` — per-stage timings as metrics (library mode)
  - `with mlprof.profile(mlflow_run=...)` — context manager that does all
    of it automatically inside a training script
  - Pre-flight audit gate that can `raise` and fail the MLflow run with
    a clear blocker reason visible in the UI
  - Predicted-vs-actual comparison built in (when both predicted and
    actual cost end up on the same run, expose the delta as a tag)

  Pros: mlprof becomes native to the MLflow UI; every value the tool
  produces is visible where users already live; predicted-vs-actual
  feedback loop makes the tool self-validating. Cons: more surface to
  maintain; more places MLflow changes can break us.

- **C. MLflow under the hood, no user-visible MLflow APIs.** Users still
  write specs; mlprof transparently logs to MLflow if it's configured.
  Pros: zero user effort. Cons: surprising magic; hard to debug; the
  user is best served seeing the integration explicitly.

### Recommendation

**B.** The predicted-vs-actual feedback loop alone is worth the surface
area. The other primitives compose nicely with library mode (decision 2)
— `mlprof.profile(mlflow_run=...)` is the unified entry point that uses
all of them under the hood, so users don't have to call each one
themselves.

### What changes if you go B

- Issue [#17](https://github.com/dnewcome/mlprof/issues/17) grows
  significantly — should be re-scoped or replaced with a richer issue
  (or split into 3-4: read-side, write-side, library-mode integration,
  pre-flight gate).
- New issue specifically for the **predicted-vs-actual feedback loop**
  as a measurable design goal — it's the killer differentiator and
  deserves explicit tracking.
- `mlflow` as a `[mlflow]` extra in pyproject (already proposed in #17).
- Optional but valuable: the AG News demo gets an MLflow-tracked variant
  to demonstrate the integration end-to-end.

## Cross-cutting: how does this resolve the dependency story

If you go B/B/B above:

| Scenario | mlprof workflow |
|---|---|
| Local Mac dev only, no training here | Install mlprof in dev venv; write spec by hand or via skill; run `audit()` only |
| Single venv with model deps + mlprof | `pip install mlprof[mlflow]`; do everything (probes, library mode, MLflow logging) |
| Docker training image | Add `mlprof[mlflow]` to image deps; use library mode inside the container; predictions log to the same MLflow run |
| Two-env (Mac dev + Docker training) | Dev venv: audit-only against MLflow runs. Container: library mode + MLflow logging. Best of both. |

The dependency wall disappears for most users because mlprof gets
installed alongside the existing training env (whichever env that is),
not into its own isolated venv.

## What's already filed vs. what would need filing

### Already filed (relevant to these decisions)
- [#1](https://github.com/dnewcome/mlprof/issues/1) Docker launcher — still relevant, less central if library mode wins
- [#7](https://github.com/dnewcome/mlprof/issues/7) YAML spec loading — orthogonal, still useful
- [#13](https://github.com/dnewcome/mlprof/issues/13) Stage-level profiling — *becomes the highest-value issue* in library mode
- [#15](https://github.com/dnewcome/mlprof/issues/15) Model size — orthogonal, still useful
- [#16](https://github.com/dnewcome/mlprof/issues/16) sagebaker ↔ mlprof relationship — needs minor update if mlprof goes MLflow-first
- [#17](https://github.com/dnewcome/mlprof/issues/17) MLflow integration — needs to be re-scoped or split if decision 3 lands on B

### Would need filing if B/B/B
- Library mode v1: promote `measure()`/`audit()` to public API + docs
- Library mode v2: `mlprof.profile()` top-level context manager
- Pre-flight audit gate (capability to `raise` on incompatibilities)
- Predicted-vs-actual feedback loop as an explicit feature
- README rewrite for MLflow-first framing
- AG News demo MLflow variant

### Would need updating
- README (headline + framing)
- generate-modelspec skill (MLflow input as primary)
- Issue #17 (re-scope to richer integration)

## Recommended path if you don't want to think harder

Commit to **B, B, B** (MLflow-first with protocol fallback; both modes;
native MLflow integration). The pieces compose. The protocol stays as
escape valve. Re-scope #17, file the four library-mode/MLflow issues
above, then implement in this order:

1. **README + framing change** (an afternoon — no code, just docs)
2. **Promote `measure()` and `audit()` to public API** (small, immediate
   user value: people can start instrumenting today)
3. **`from_mlflow_run()` helper** (cuts skill complexity, removes the
   biggest manual-spec friction)
4. **`mlprof.profile()` context + MLflow logging** (the unified entry
   point users will actually use)
5. **Stage profiling [#13](https://github.com/dnewcome/mlprof/issues/13)**
   inside the library-mode + MLflow context
6. **Predicted-vs-actual** once both predictions and library-mode
   measurements are landing on MLflow runs

That sequence builds incremental value and ends with the tool that
matches the discussion's eventual shape.

## Things I'd push back on if you proposed them

- **Going MLflow-only (C on decision 1)** — irreversible, throws away
  the protocol which costs nothing to keep. Even if you only ever use
  MLflow yourself, leaving the agnostic path means contributors with
  W&B / Neptune / homegrown trackers can extend later.
- **Library-only (C on decision 2)** — the scaling extrapolation + Pareto
  frontier IS the original value prop. Probe mode is what makes mlprof a
  planning tool, not just a measurement library.
- **Magic MLflow (C on decision 3)** — implicit cross-cutting integrations
  are hard to debug when they break. Users are best served by explicit
  primitives even if a top-level convenience wrapper exists.

## Things this plan does *not* decide

- Whether to support W&B / Neptune / other trackers. Not now; could be
  added behind a tracker-abstraction later if you ever want.
- Whether to publish to PyPI. Not now; install from git is fine for an
  early project.
- Whether the project needs a separate website / hosted docs. Not now;
  README + per-issue specs are enough.
- What to do about sagebaker integration beyond
  [#16](https://github.com/dnewcome/mlprof/issues/16). The
  `from_sagebaker_plugin()` adapter remains optional.
