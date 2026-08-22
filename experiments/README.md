# Experiment runners

This directory contains experiment orchestration only.  Core learning logic
remains in `src/llm_modulo_cegis/`, and the shared training CLI remains
`../run_obstacle_avoid.py`.

Current gated experiment:

- `run_carrywater_q48_multiseed.py`: matched Q48 correct-composite-only versus
  full-bank protocol.  Run `--validate-only` before launching training.

Historical, reproducible ablations:

- `run_ablation.py`: semantic closed loop versus frozen revisions.
- `run_falsifier_multiseed.py`: single radius versus radius ladder.
- `run_numeric_fitting_multiseed.py`: query bootstrap versus full buffer.
- `run_violation_pooling_multiseed.py`: violation-credit assignment.
- `run_linear_max_support_gate_multiseed.py`: support-gate audit/enforcement.

Moving these files changed only their repository paths.  Existing outputs and
their implementation manifests were deliberately left untouched.
