# LLM-Modulo Semantic–Numeric CEGIS

This is a second implementation, independent of the older `llm_guided_cegis`. It does not modify `icrl/pucl.py` or `icrl/constraint_net.py`, and it does not read any state-level training labels beyond the expert data.

The core change: the LLM no longer emits a single configuration once at the start of training. Instead it acts as the hypothesis synthesizer of an outer semantic CEGIS loop, while the neural network, the trajectory optimizer, and the trajectory-level oracle form an inner numeric CEGIS loop. Every round of numeric learning produces structured evidence, and the LLM must use it to retain, retire, replace, or split hypotheses, or to design the next intervention.

## Data flow

```text
Task text + variable schema
          │
          ▼
 Qwen: generate candidate hypothesis population H1...HK
          │  strict JSON/IR compilation
          ▼
 One independent neural constraint head per hypothesis
          │
          ├── multiple-instance trajectory-level training
          ├── hypothesis-specific differentiable falsifier
          └── whole-trajectory safe/violation oracle
          │
          ▼
 EvidenceCompiler: accuracy, counterexamples, intervention hit rate, uncertainty
          │  no ground-truth geometry, state labels, or IoU
          ▼
 Qwen: retain / retire / replace / split / intervention
          └─────────────────────────────── next outer round
```

The final obstacle boundary is still learned by the neural network. What the LLM decides is the constraint search space and the follow-up experiments; it never outputs centers, radii, thresholds, or boundary points.

## Implemented capabilities

- A hypothesis population and version space, rather than a single irrevocable hypothesis;
- Variables: `x_position/y_position/x_velocity/y_velocity/speed/progress`;
- Both joint and independent coupling;
- `forbidden_region` supports an implicit MLP as well as a linear neural control head;
- Univariate `upper_bound/lower_bound` uses learnable threshold heads;
- `max/mean/last` trajectory aggregation, where the choice changes the computation graph;
- Trajectory-level multiple-instance learning with latent witnesses;
- Ensemble epistemic uncertainty;
- Hypothesis-specific differentiable falsification;
- Qwen reads structured evidence each round and returns typed revision actions;
- Compilation failures, illegal variables, illegal relations, and contradictory actions are all explicitly logged and safely rolled back;
- When Qwen output is truncated, only JSON objects that are already closed and pass compilation are recovered and accepted; malformed objects are never executed;
- Qwen prompts, raw outputs, parse errors, fallbacks, hypothesis versions, and all oracle queries are auditable;
- The hidden circular ground truth is accessed only by the experiment harness's oracle and the final evaluator;
- Ablation scripts for the closed loop and for a frozen one-shot bank.

## Layout

```text
llm_modulo_cegis/
├── configs/
│   ├── obstacle_avoid_smoke.yaml
│   ├── obstacle_avoid_qwen_smoke.yaml
│   ├── obstacle_avoid_qwen_test.yaml
│   └── obstacle_avoid_qwen.yaml
├── src/llm_modulo_cegis/
│   ├── data.py          # shared data, variable schema, differentiable derived features
│   ├── hypotheses.py    # IR, compiler, hypothesis bank, revision actions
│   ├── semantic.py      # Qwen and the deterministic ablation reasoner
│   ├── learner.py       # multi-hypothesis neural model and trajectory-level MIL
│   ├── falsifier.py     # hypothesis-specific counterexample trajectory optimization
│   ├── evidence.py      # numeric results → leakage-free semantic evidence
│   ├── oracle.py        # capability-limited whole-trajectory binary interface
│   ├── evaluation.py    # ground-truth metrics and plots, evaluation only
│   └── loop.py          # bi-level closed-loop controller
├── tests/test_core.py
├── run_obstacle_avoid.py
└── run_ablation.py
```

## Running

The project's designated `pucl` environment can run the non-LLM closed loop and the tests:

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe -m unittest discover -s llm_modulo_cegis\tests -v

C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_smoke.yaml
```

The `pucl` environment currently does not have `transformers` installed. The machine's Python 3.12 has the dependencies Qwen needs, so local Qwen tests use:

```powershell
py -3.12 llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_qwen_smoke.yaml

py -3.12 llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_qwen.yaml
```

The full configuration is computationally heavy. On CPU, run the smoke config first; `obstacle_avoid_qwen.yaml` trains 3-member ensembles over 4 outer rounds and performs multiple trajectory optimizations.

Ablation:

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\run_ablation.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_smoke.yaml
```

## Outputs and auditability

Every run produces:

- `semantic_interactions.json`: all prompts, raw Qwen outputs, parse errors, and final actions;
- `hypothesis_bank.json`: the admission, retention, and retirement history of each hypothesis;
- `evidence_history.json`: the leakage-free evidence actually fed back to the LLM;
- `evaluation_history.json`: ground-truth evaluation kept in isolation, never fed back to the LLM;
- `oracle_queries.npz` and `oracle_query_log.json`: whole-trajectory queries;
- `query_diagnostics.json`: falsifier objectives and per-hypothesis predictions before each query;
- `constraint_models.pt`: all hypothesis models plus the final champion;
- `learned_boundary.png`: the final neural boundary;
- `semantic_trace.png`: hypothesis scores and semantic actions;
- `result.json`: the final summary.

## Current limitations

1. `ObstacleAvoid` has only two-dimensional observations, so the evidence for structure identification is weak; a formal paper must add distractor dimensions and heterogeneous constraints.
2. The oracle returns whole-trajectory labels only. Violation timing is still estimated by the MIL latent witness and must not be treated as ground-truth localization.
3. Qwen2.5-1.5B can generate initial hypotheses and revise them from evidence, but its ability to follow complex JSON is still limited; the system rejects illegal objects item by item and only adds deterministic candidates when there are too few legal ones.
4. The smoke configuration only verifies the closed loop; it says nothing about boundary quality. Use the full training schedule for evaluation.
5. The current model family implements MLP, linear, and scalar threshold heads. `GP/SDF/symbolic formulas` can be added through the compiler interface, but are not pretended to be supported already.

For the detailed mathematics and implementation boundaries see [ARCHITECTURE.md](ARCHITECTURE.md); for baseline measurement records see [reports/IMPLEMENTATION_AND_TEST.md](reports/IMPLEMENTATION_AND_TEST.md); for the Qwen closed-loop test on ObstacleAvoid see [reports/OBSTACLE_AVOID_QWEN_TEST.md](reports/OBSTACLE_AVOID_QWEN_TEST.md).
