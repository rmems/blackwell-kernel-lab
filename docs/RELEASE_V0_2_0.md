# v0.2.0 — First Agoge CUDA operator: MiniCPM5 + Granite

This release turns the lab's measurement foundations into a first-party CUDA
operator consumed by agoge-forger on the RTX 5080 (sm_120, ~16 GB). It requires
a repeatable training-speed improvement on **both** MiniCPM5 and Granite 4.1.
The operator is selected from measured training evidence, not from a target
CUDA language percentage.

## Delivery sequence

Canonical planning is in Linear; all new issues originate there. Keep existing
GitHub mirrors aligned, and link implementation PRs to the exact Linear issue.
Parent: [RM-175](https://linear.app/rpd-34/issue/RM-175). Release:
[GitHub milestone #8](https://github.com/rmems/blackwell-kernel-lab/milestone/8).

| Issue | Deliverable | Gate |
|---|---|---|
| [RM-1770](https://linear.app/rpd-34/issue/RM-1770) | PR-only main, required CPU/conditional GPU checks, release policy | Verify the new check before enabling the rule |
| [RM-1771](https://linear.app/rpd-34/issue/RM-1771) | Profile both models; freeze one operator contract | Existing sampler, live marker/profiling acceptance, model qualification |
| [RM-1772](https://linear.app/rpd-34/issue/RM-1772) | CUDA implementation, bindings, package, correctness and operator benchmark | Accepted measured gap and interface from RM-1771 |
| [RM-1773](https://linear.app/rpd-34/issue/RM-1773) | Version-pinned opt-in Agoge consumer | BKL package and contract |
| [RM-1774](https://linear.app/rpd-34/issue/RM-1774) | Both-model qualification, release artifacts, v0.2.0 tag | Consumer, PR/CI enforcement, and every release gate below |

[RM-1228](https://linear.app/rpd-34/issue/RM-1228) is the supporting telemetry
epic, including existing sampler [RM-1230](https://linear.app/rpd-34/issue/RM-1230).
[Agoge PR #147](https://github.com/rmems/agoge-forger/pull/147) already merged
marker and profiling hooks. Reuse them; merged CPU-tested implementation is
distinct from verified live-GPU acceptance in RM-1231/RM-1233.

Qualification [RM-779](https://linear.app/rpd-34/issue/RM-779) remains Agoge-owned.
Draft experiment claims do not substitute for accepted workload provenance.
Inference-ablation and Green Context maintenance are outside this release's
critical path unless the selected operator demonstrates an actual dependency.

## Operator and ownership contract

Before implementation, RM-1771 names the selected operation and freezes its
supported shapes, strides, dtypes, forward/backward semantics, numerical
tolerances, memory/stream behavior, reference, calling interface, and tests.
Prefer one shared operator with a measured gap in both workloads. If no
candidate is defensible, record a no-go and keep downstream delivery blocked.

BKL owns device code, bindings, and the versioned PyTorch-compatible package.
Agoge owns training, model loading, datasets, and lifecycle. No `.cu` copies go
into Agoge. The consumer remains explicit opt-in and preserves reference
execution. Qualification must record actual BKL execution; a silent reference
fallback cannot count as device evidence.

## Release acceptance

Use `openbmb/MiniCPM5-1B-Base` and `ibm-granite/granite-4.1-3b-base`, with exact
qualified model/tokenizer revisions and frozen data/configs. Preserve the same
accepted batches, objective, and update schedule between reference and BKL.

1. Validate outputs and gradients where applicable against tolerances frozen
   before optimization. Preserve training loss behavior, adapter save,
   clean-process reload, and deterministic generation smoke.
2. Run three independent reference/BKL pairs **per model**, alternating order.
   Each run has at least five warmup optimizer steps and thirty measured steps.
   Every pair, on both models, must show **at least 5% lower median steady-state
   optimizer-step time**. Retain raw timings, throughput, and spread.
3. Separate bounded profiling runs from performance runs. Record profiler and
   collector overhead, both source SHAs, package/build identity, environment,
   immutable workload pins, and actual backend execution.
4. Target sm_120; no sm_100, FA4, TMEM, or MIG assumptions. Maintain **at least
   2048 MiB free throughout** each measurement, not only before allocation.
   Serialize training and GPU CI. Missing reserve or competing workloads
   invalidate the measurement. The headroom probe is not an exclusive lock.
5. Keep raw measurements in gitignored `results/`. Publish compact reproducible
   evidence, install/reproduction instructions, and package artifacts. Required
   checks and review requirements must pass before release delivery.

Failed, inconclusive, or no-go model results leave v0.2.0 open. Do not lower
the thresholds after seeing results or claim training gains from synthetic
microbenchmarks. Full model comparisons are release evidence, not mandatory
training runs on every PR.

## Milestones and release history

New release milestones include their version and close only when the matching
tag and release artifacts exist and have been verified. v0.2.0 is an upcoming
release target, not a published package or a completed performance claim.

The completed M1/M2/K0/CI/K1 work tracks were retired on 2026-09-22 as an explicit
historical exception. Their issue membership remains intact; closure creates
no retrospective tags. Existing v0.1.0 history and retired M3 remain intact.
Linear's former agent milestones are historical, and myelin M4–M6 tracks are
retired external work. They do not form a release ladder for this backend.
