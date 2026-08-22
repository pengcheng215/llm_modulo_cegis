# GPT proxy-guarded implementation and test (2026-08-17)

## Outcome

The final live run selected the intended qualitative structure: a nonlinear
forbidden region over `(x_position, y_position)`. GPT produced valid structured
outputs throughout; no semantic fallback or augmentation was used.

Final evaluation-only metrics (never exposed to the learner or LLM):

| metric | value |
|---|---:|
| IoU | 0.5827 |
| accuracy | 0.9537 |
| false-safe rate | 0.1478 |
| false-unsafe rate | 0.0380 |
| held-out expert safe rate | 0.6667 |
| Oracle queries | 26 |
| GPT interactions / fallbacks | 3 / 0 |

Artifacts are in `outputs/gpt_proxy_guarded/`.

## What was implemented

### 1. Simple hypotheses retain a real selection opportunity

- A strict feature-superset hypothesis cannot be champion unless its
  prequential balanced accuracy improves over the smaller hypothesis by at
  least `nested_minimum_balanced_accuracy_gain` (0.08 in this run).
- A hypothesis containing a time/progress variable cannot use it as a proxy for
  a physical state coordinate when a comparable physical-state hypothesis
  exists, unless it gains at least 0.08 balanced accuracy.
- Intervention violation yield now has weight 0.03 rather than 0.15 in champion
  selection. Generating many violations measures acquisition usefulness; it is
  not strong evidence that the source hypothesis is semantically correct.
- Composite typed clauses remain supported for genuine simultaneous
  constraints, such as an equality band on A and an upper bound on B.

In the final run, `h_phase_lateral` had raw selection score 0.633 and balanced
accuracy 0.900. It was nevertheless rejected with
`task_progress_proxy_without_material_evidence_gain`, because the physical
`h_spatial_nonlinear` reached balanced accuracy 0.850 without using progress.
The latter became the qualified champion.

### 2. False-unsafe falsification has a hard trust region

For a `model_false_unsafe` intervention, every optimized point is projected to
remain within Euclidean radius 0.08 of the corresponding expert state. The
projection runs initially and after every optimizer step. A final validator
rejects any trajectory exceeding the radius. This prevents the purported
"near-expert safe" query from drifting into an unrelated heuristic path.

Future `query_diagnostics.json` files record both the configured trust radius
and the measured maximum pointwise deviation for each candidate.

### 3. Oracle labels are partitioned by role

The 26-query budget is unchanged:

- 14 warm-up queries;
- 4 queries in each of three outer rounds.

Within warm-up, labels are partitioned into gradient training, structure audit,
and final threshold calibration. The four calibration examples (2 safe, 2
violation) are never used by gradient descent. All outer-loop answers are still
shared across every active hypothesis via frozen pre-query predictions.

### 4. Final refitting is transactional

After a champion qualifies, its structure is frozen and a candidate is
reinitialized and trained using all nine training experts (five fitting plus
four structure-audit experts) and all eligible query labels. A scalar decision
threshold is then chosen on the disjoint calibration subset, subject to the
training-expert safety constraint.

The candidate is committed only if calibration safe accuracy, violation recall,
balanced accuracy, expert safety, and non-regression against the incumbent all
pass. Otherwise the registry restores the exact incumbent model object.

This guard was necessary. In the final run, the refitted candidate classified
zero of two calibration violations correctly. It was rejected for:

- `calibration_violation_recall_below_gate`;
- `calibration_balanced_accuracy_below_gate`;
- `calibration_worse_than_incumbent`.

Consequently, `finalization_applied=false`, the decision threshold stayed at
0.0, and the final evaluation metrics exactly equal the pre-finalization ones.
The earlier unguarded experiment (`outputs/gpt_finalized/`) demonstrates why
this matters: it improved held-out expert safety to 1.0 but collapsed IoU from
0.1899 to 0.0686 and accuracy to 0.5510.

## Where the earlier error occurred

GPT did not fail to produce the spatial hypothesis. In repeated runs it proposed
the correct `(x, y)` nonlinear region alongside phase-dependent alternatives.
The failure occurred in model selection: progress is correlated with horizontal
motion, so a `(y, progress)` MLP can imitate `(x, y)` on the queried trajectories.
Its high intervention yield then outweighed the simpler model's better or nearly
equal predictive evidence. The new proxy-variable gate and lower intervention
yield weight remove that path to a false semantic conclusion.

This also explains the earlier "the expert moves upward, therefore y must stay
high" class of error. A demonstration shows a correlated behavior, not which
variable causally defines feasibility. Above- and below-obstacle demonstrations
eliminate a global high-y rule; the progress proxy guard additionally prevents a
time variable from reconstructing obstacle location indirectly.

## Verification

- Python compilation passed for all modified modules.
- 28 unit tests passed, including strict structured output, composite
  constraints, prequential evidence, rolling windows, hard-max inference,
  decision calibration, hard false-unsafe projection, nested-feature minimality,
  and progress-proxy rejection.
- The final authorized GPT run completed in 136.7 seconds with no fallback.

## Remaining limitation

The selected boundary is qualitatively correct, but held-out expert safe rate is
still 0.6667. The attempted from-scratch all-expert refit was too unstable to
commit. The next controlled improvement should compare incumbent fine-tuning and
multiple deterministic refit restarts, selecting only through a larger disjoint
calibration set. Until that experiment succeeds, the system correctly reports
the limitation instead of trading broad unsafe-area accuracy for expert safety.
