# SemTraj2D v1.1.0

SemTraj2D is a controlled benchmark for semantic constraint identification from
three learner-visible information channels: a natural-language task
description, safe expert trajectories, and a budgeted whole-trajectory
membership oracle.  It is designed to test mechanisms that the original
single-circle Obstacle2D smoke task cannot distinguish: proxy features,
simple-versus-capacity-heavy structures, dynamic constraints, heterogeneous
compositions, counterfactual generalization, and abstention outside the current
hypothesis language.

This directory is a development release generated with public sampling seed
`20260821` and a separate non-exported private root seed.  It is suitable for
implementation testing and preregistering the evaluation protocol.  A paper result still requires the fixed multi-instance
experiment and baselines described below; this one generated release is not by
itself evidence that the proposed architecture is superior.

## Information contract

During learning, mount only `public/<task>/` and expose the binary membership
oracle as a capability.  Do not give learner or semantic reasoner direct file
access to `private/`.

Public information consists of:

- `task_spec.json`: description, workspace, horizon, maximum step, and feature
  meanings, without a numerical constraint boundary;
- `expert_trajectories.npz`: 30 safe demonstrations of shape `[100, 2]`;
- `splits.json`: deterministic `18/6/6` train/validation/test expert IDs;
- `manifest.json`: hashes and shape metadata for public artifacts.

Private information consists of:

- `oracle.json`: the analytic rule used only by the membership service and
  post-hoc evaluator; it does not contain expected structure;
- `expected_structure.json`: post-hoc typed structure truth;
- `evaluation_trajectories.npz`: balanced, globally constructed test probes;
- `manifest.json`: private hashes and per-stratum counts.

The private probes are sampled from cubic Bezier control points over the full
workspace.  They do not copy or perturb expert interior waypoints.  Spatial
collision labels use continuous polyline-to-obstacle distance, so a segment
cannot jump through an obstacle between sampled states.

## Feature library

All hypotheses choose from the same 12 differentiable features derived from a
two-dimensional path:

| Group | Features |
|---|---|
| position | `x_position`, `y_position` |
| velocity | `x_velocity`, `y_velocity`, `speed` |
| acceleration | `x_acceleration`, `y_acceleration`, `acceleration_norm` |
| direction | `heading_sin`, `heading_cos` |
| path history | `path_length_so_far` |
| time | `progress` |

Only one to three features are causal in a task.  The remaining features are
correlated distractors that make an unrestricted all-feature MLP meaningfully
different from typed semantic structure selection.

## Tasks

| Task | Hidden structure | What it tests | Private test size |
|---|---|---|---:|
| `disk_clean` | joint `(x,y)` keep-out region | spatial sanity check with upper and lower experts | 384 |
| `disk_upper_proxy` | exactly the same hidden rule as `disk_clean` | causal/proxy error when all public experts happen to go above | 384 |
| `diagonal_halfspace` | joint linear positional boundary | whether a simple linear structure has room to win | 384 |
| `lane_band` | scalar `y` equality band | feature-specific equality rather than a broad joint MLP | 384 |
| `speed_limit` | scalar speed upper bound | a dynamic constraint for which static position IoU is invalid | 384 |
| `disk_and_speed` | spatial keep-out OR speed upper bound | heterogeneous clauses and isolated clause violations | 672 |
| `lane_and_speed` | scalar equality band OR speed upper bound | the advisor's feature-A equality plus feature-B inequality case | 672 |
| `eventually_visit_open_set` | must visit a checkpoint | negative control outside the current max/mean/last violation IR | 384 |

`disk_clean` and `disk_upper_proxy` have byte-identical descriptions, private
rules, evaluation arrays, expert endpoints, expert x curves, trajectory IDs,
and splits.  For half of the matched experts, only the detour side is reflected
from upper to lower.  This paired design measures sensitivity to demonstration
style without confounding it with geometry, endpoints, or test difficulty.

Each atomic private bank contains equal safe and violation counts in four
groups: `id`, `boundary`, `counterfactual`, and `ood`.  Counterfactual pairs
share exactly the same endpoints and differ by one controlled path change.  The
spatial composite additionally contains `spatial_only`, `speed_only`, and
`multi_clause` strata.  The equality/inequality composite analogously contains
`lane_only`, `speed_only`, and `multi_clause`.  Exact per-clause labels remain
private.  Speed probes are direction-balanced: the true speed score perfectly
separates the generated labels, while no signed single-axis velocity reaches a
best threshold balanced accuracy of 0.90.

## Generation and validation

The generator refuses to overwrite an existing suite.  Generate another fixed
instance into a new directory:

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe `
  llm_modulo_cegis\tools\generate_semtraj2d.py `
  --output llm_modulo_cegis\data\SemTraj2D_seed_20260822 `
  --public-seed 20260822 `
  --expert-count 30 `
  --per-group-label-count 48
```

If `--private-seed-file` is omitted, the generator uses a cryptographically
random 256-bit private seed and does not export it.  A controlled evaluation
service may instead provide a secret hexadecimal seed file of at least 128
bits.  Never derive the private seed from the public seed or store it in a
learner-visible manifest.

Run the complete legacy-plus-benchmark regression suite:

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe -m unittest discover `
  -s llm_modulo_cegis\tests -v
```

The automated contract checks verify all expert labels, split disjointness,
balanced private classes, identical clean/confounded test banks, matched
counterfactual endpoints, isolated composite violations, proxy resistance of
the speed task, exact TaskSpec keys, continuous segment collision,
differentiable features, tie-aware rank metrics, structure matching, and
deterministic NPZ bytes.

For an official run, keep `data.inline_private_evaluation: false`.  Training
then writes a `freeze_manifest.json` that seals the selected champion and model
hash before the private trajectory archive is opened.  Evaluate in a second
process:

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe `
  llm_modulo_cegis\tools\evaluate_semtraj2d.py `
  --run-dir outputs\semtraj2d_disk_upper_train `
  --private-dir llm_modulo_cegis\data\SemTraj2D\private\disk_upper_proxy
```

Do not use `--diagnostic-all-hypotheses` for an official score: that switch is
only for development and exposes test performance for non-selected models.

## Primary metrics

Use private trajectory metrics as the main result:

- balanced accuracy and violation AUPRC;
- safe accuracy and violation recall, reported together;
- worst-group balanced accuracy over ID, boundary, counterfactual, and OOD;
- per-group balanced accuracy;
- Oracle calls and area under performance-versus-query-budget curve;
- permutation-invariant structure match, including every composite clause;
- qualified exact structure recovery, which additionally requires the frozen
  champion to have passed all training-time evidence gates;
- correct `inconclusive` rate on the open-set task.

Static state-grid IoU is secondary and is intentionally undefined for dynamic
and heterogeneous tasks.  A learner can have acceptable average accuracy while
failing every safe OOD route, so average ID accuracy alone is not sufficient.

## Minimum paper protocol

Use at least five preregistered hidden task instances.  For each task instance,
freeze each semantic hypothesis bank once, then replay five numeric seeds so
semantic and optimizer variance are not mixed.  Keep task-instance seed,
semantic seed, and numeric seed as separate columns in every result row.

At the same Oracle and candidate-compute budgets, include at minimum:

1. all-feature joint MLP CEGIS;
2. typed exhaustive CEGIS without an LLM;
3. one-shot frozen LLM bank;
4. the complete LLM-modulo CEGIS loop;
5. an oracle-structure/no-numeric-boundary upper bound.

The central paired result is the change from `disk_clean` to
`disk_upper_proxy`, especially safe OOD accuracy and worst-group balanced
accuracy.  The central composite result reports recall separately for
spatial-only and speed-only violations.  The open-set result prevents selective
reporting only on tasks guaranteed to fit the method's hypothesis language.

The design follows the trajectory-level constraint-learning evaluation setting
used by the [ICRL benchmark](https://arxiv.org/abs/2206.09670), and uses active
counterfactual strata because passively observed expert correlations can cause
causal confusion, as studied in
[Gupta et al.](https://proceedings.mlr.press/v213/gupta23a.html).  Multiple
geometry environments are needed because robustness to a single training
environment does not establish invariant structure recovery; see
[Ahuja et al.](https://proceedings.mlr.press/v119/ahuja20a.html).

## Current scope

The checked-in bundle is still a development set because both public and
private files are locally available.  A final blind paper test should be newly
generated after code/configuration freeze, held by an evaluator service, and
scored with limited submissions.  SemTraj2D is a controlled mechanism
benchmark, not a replacement for an external-validity robotics experiment.
For publication, pair it with at least one higher-dimensional simulator task
whose trajectories, dynamics, and Oracle come from a source independent of
this hypothesis IR.  Do not claim that the architecture is generally better
from the development seed or smoke profile.

## Development-seed validation snapshot

The authorized GPT semantic checks on `disk_upper_proxy` and `lane_and_speed`
used the public bundle only.  They accepted `3` and `2` hypotheses respectively,
with zero fallback or fallback augmentation.  Their first hypotheses matched
the intended disk structure and the intended equality-plus-inequality
composition.

The complete legacy-plus-benchmark regression suite currently passes all 99
tests, including the public/private contract, paired-task invariants, proxy
audits, continuous collision checks, structure scoring, and post-hoc metrics.

The current two-round disk closed-loop smoke used 26 membership queries.  Its
frozen candidate has an exact typed-structure match, but it did not pass the
safe-accuracy qualification gate.  Independent post-hoc evaluation gave
trajectory balanced accuracy `0.640625`, safe accuracy `0.28125`, violation
recall `1.0`, and worst-group balanced accuracy `0.5`.  Thus semantic grounding
worked in this smoke, while numeric fitting rejected too many valid routes.
This deliberately remains a diagnostic result, not a paper comparison.  Full
details and artifact paths are recorded in
[the benchmark report](../../reports/SEMTRAJ2D_BENCHMARK_2026-08-21.md).
