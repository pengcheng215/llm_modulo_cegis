# GPT integrated final run (2026-08-17)

## Final outcome

The integrated GPT run selected the intended qualitative constraint:

`(x_position, y_position) -> joint nonlinear forbidden_region -> max over time`

Evaluation-only metrics:

| metric | value |
|---|---:|
| IoU | 0.8146 |
| accuracy | 0.9855 |
| false-safe rate | 0.1613 |
| false-unsafe rate | 0.00243 |
| held-out expert safe rate | 1.0000 |
| Oracle queries | 26 |
| GPT interactions / fallbacks | 3 / 0 |

The champion was the only hypothesis that passed all qualification gates. Its
prequential balanced accuracy was 0.900 and selection score was 0.637.

Artifacts are stored in `outputs/gpt_integrated_final/`.

## Changes included in this run

### Intervention-localized witnesses

The previous latent-witness heuristic selected states merely because they were
far from the expert-state cloud. This could confuse sparse but safe corridors
with constraint violations.

The replacement is tied to the actual intervention:

- warm-up deformations store the timestep of maximum displacement from the
  source expert;
- every optimized falsifier stores the source model's pre-Oracle argmax;
- a source-model argmax is used as a weak witness only for a
  `model_false_safe` trajectory, only after the Oracle labels the entire
  trajectory as violation, and only for the source hypothesis;
- all other hypotheses receive only the shared whole-trajectory label.

Thus the Oracle still provides no violation timestamp. The stored witness is an
auditable optimizer attribution rather than hidden geometry.

### Position versus derived-proxy protection

Time/progress and velocity can identify that a trajectory was modified without
representing the persistent spatial constraint. Qualification therefore
requires:

- a progress-bearing alternative to beat a comparable physical-state model by
  at least 0.08 balanced accuracy;
- a pure velocity alternative to beat a comparable position model by at least
  0.20 balanced accuracy.

This is a task-level causal prior for persistent obstacle experiments and is
configurable. It does not inspect the private circle parameters or IoU.

### Falsifier trust region

All `model_false_unsafe` trajectories are projected pointwise to radius 0.08
around the source expert, both initially and after every optimizer step. In this
run all 12 logged false-unsafe candidates respected the bound; the largest
measured deviation was 0.08000033.

### Transactional finalization with a sample-size gate

The code can compare an incumbent calibration, incumbent fine-tuning under
several sparse-region priors, and deterministic scratch restarts. Calibration
and candidate submission use disjoint label subsets.

However, this run had only two safe and two violation trajectories in the
submission holdout. Selecting among eight candidate models with four labels was
shown by replay to overfit. Finalization now requires at least three examples of
each class in the submission holdout. It therefore exited before candidate
training with:

`insufficient_selection_holdout_examples_per_label`

The qualified incumbent was preserved exactly (`finalization_applied=false`).
This retained held-out expert safety of 1.0.

## Stage diagnosis

The correct spatial model reached IoU 0.700 after round 2 and 0.815 after round
3. Competing affine, speed, and terminal-band hypotheses all failed violation
recall. GPT produced four valid initial hypotheses, used no fallback, and did
not require hypothesis-bank augmentation.

The global Oracle budget remained fixed: 14 warm-up queries plus four selected
queries in each of three outer rounds. Candidate generation for unselected
hypotheses did not query the Oracle.

## Verification

- 32 unit tests passed.
- All modified Python modules compiled successfully.
- The real GPT run completed in 112.6 seconds.
- No API key was logged or written to artifacts.
- `icrl/pucl.py` and `icrl/constraint_net.py` were not modified.

## Remaining limitations

- The Oracle labels were imbalanced (6 safe, 20 violation). The fixed budget
  still needs a better safe-query acquisition mechanism for other tasks.
- Dynamic-proxy thresholds are semantic priors and should be ablated across
  tasks with genuine velocity constraints.
- Final multi-candidate refitting is intentionally disabled at the current
  sample size. Enabling it without increasing Oracle calls would require a
  carefully designed cross-fitting experiment rather than reusing the same
  four labels.
