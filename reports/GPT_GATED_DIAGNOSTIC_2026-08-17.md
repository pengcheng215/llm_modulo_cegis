# GPT gated diagnostic — 2026-08-17

## Faults fixed

1. Training used smooth-max while evidence and final evaluation disagreed about
   hard max semantics. Smooth-max is now training-only; inference, acquisition,
   evidence, and evaluation all use the hard trajectory predicate.
2. A weighted score could declare a model with poor safe accuracy champion.
   Qualification now requires enough frozen labels, safe/violation class gates,
   fit-expert consistency, and structure-audit expert consistency.
3. When no model qualifies, the result is `inconclusive`; semantic actions may
   query but may not prune or replace the bank.
4. A malformed/truncated revision uses a conservative query-only fallback and
   cannot retire hypotheses.
5. Prompts explicitly state that temporal `max` means any positive per-state
   violation, not the maximum raw feature value.
6. Candidate pools include false-safe and false-unsafe probes. A pending LLM
   intervention no longer duplicates itself across every pool slot.
7. Selection uses the fixed warmup audit plus the latest two outer rounds of
   frozen predictions. Full prequential history remains in diagnostics, but an
   obsolete early model version no longer permanently determines the current
   structure.

## Final command

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe .\llm_modulo_cegis\run_obstacle_avoid.py `
  --config .\llm_modulo_cegis\configs\obstacle_avoid_gpt_diagnostic.yaml `
  --output .\llm_modulo_cegis\outputs\gpt_diagnostic_rolling --overwrite
```

The actual response model was `gpt-5.6-sol`. Initial synthesis and both
actionable revision rounds completed with zero fallback and zero augmentation.

## Stage behavior

- Round 1: no hypothesis qualified; the bank was preserved.
- Round 2: no hypothesis qualified; the joint planar MLP became the provisional
  query target and already reached evaluation-only IoU `0.637`.
- Round 3: the joint planar MLP became the only qualified hypothesis.

Final selection evidence for the joint model:

- safe accuracy: `0.8571`;
- violation recall: `0.7778`;
- fit-expert safe rate: `1.0`;
- structure-audit expert safe rate: `1.0`;
- balanced accuracy: `0.8175`;
- prequential observations in the selection window: `16`.

Competing affine and speed hypotheses had violation recall `0`; the independent
position hypothesis failed all four qualification gates.

## Isolated evaluation

- IoU: `0.7912`;
- grid accuracy: `0.9827`;
- false-safe rate: `0.1344`;
- false-unsafe rate: `0.0077`;
- held-out expert trajectory safe rate: `0.6667`;
- Oracle queries: `26` (`9` safe, `17` violation).

The result is a substantial improvement over the earlier IoU `0` scalar-floor
shortcut and is now reported as `selection_status=qualified`. It is not a solved
benchmark: two of six isolated held-out expert trajectories are still rejected,
and the unsafe region misses about 13.4% of the true obstacle grid. These metrics
remain evaluation-only and are never sent to GPT or used by the selector.

## Audit artifacts

Artifacts are under `outputs/gpt_diagnostic_rolling/`:

- `stage_diagnostics.json`: data split and per-stage model snapshots;
- `query_diagnostics.json`: every selected and unselected candidate;
- `evidence_history.json`: the exact evidence visible to the semantic loop;
- `all_hypothesis_evaluation.json`: isolated evaluation of all final active models;
- `semantic_interactions.json`: prompts, raw GPT outputs, repairs, and fallback flags;
- `result.json`: final qualified result.
