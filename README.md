# LLM-Modulo Semantic–Numeric CEGIS

This is a second implementation, independent of the older `llm_guided_cegis`. It does not modify `icrl/pucl.py` or `icrl/constraint_net.py`, and it does not read any state-level training labels beyond the expert data.

The core change: the LLM no longer emits a single configuration once at the start of training. Instead it acts as the hypothesis synthesizer of an outer semantic CEGIS loop, while the neural network, the trajectory optimizer, and the trajectory-level oracle form an inner numeric CEGIS loop. Every round of numeric learning produces structured evidence, and the LLM must use it to retain, retire, replace, split, or compose hypotheses, or to design the next intervention.

The current revision also addresses five failure modes found during advisor review:

- model selection uses predictions frozen before each Oracle answer plus a pair-family-stratified warmup audit split that is excluded from gradient fitting;
- an all-feature joint MLP pays structural and parameter-count penalties; a same-family strict feature superset additionally needs the configured balanced-accuracy gain to pass nested minimality, while `simplicity_tolerance` is only a deterministic semantic-policy tie rule rather than the terminal selector;
- falsifiers first form a cheap candidate pool, after which one global acquisition rule selects only `oracle_query_budget_per_round` trajectories across promising hypotheses;
- a hypothesis can be an OR-of-violations composition of typed clauses, including heterogeneous combinations such as an equality band on feature A and an upper bound on feature B.
- an atomic 1D/2D joint linear-max `forbidden_region` hypothesis is excluded from champion selection when Oracle-violation trajectories from at least two distinct safe anchors are contained in their anchors' convex hulls, which is an exact parameter-independent contradiction for that model family.

## Architecture evolution (initial → current, updated 2026-08-21)

The first 2026-08-13 version of **this directory already had** a typed
hypothesis population, independent learners, a trajectory-membership Oracle,
and an outer loop. It was not the older one-shot `llm_guided_cegis` prototype.
In practice, however, Qwen usually fell back to or was augmented by the
canonical bank and its revisions rarely changed later numeric rounds. The
system has therefore evolved from a runnable but weakly coupled double loop
into one with strict data roles, selective acquisition, certified probes, and
structural rejection. The old descriptions and experiment numbers remain in
place as historical snapshots.

| subsystem | first `llm_modulo_cegis` behavior | current architecture | why it changed |
|---|---|---|---|
| LLM role | Qwen often produced only one broad valid candidate; canonical augmentation/full fallback supplied most of the bank, and revisions were mostly retain actions | GPT strict Structured Outputs or Qwen surface repair feeds the same trusted typed compiler; schema/semantic rejection is audited, while backend failure is fail-fast unless fallback is explicitly enabled | weak structured output must not silently become a successful semantic decision |
| hypothesis space | a typed but small and overlapping population in which a joint all-feature MLP could subsume simpler ideas | a typed, versioned population with explicit variables, relation, coupling, temporal operator, model family, and heterogeneous composite clauses | simple equalities/inequalities and feature-specific alternatives need independent falsification and selection |
| outer loop | fit/query/evidence/revise existed, but fallback hypotheses and low-impact retain actions made the semantic loop nearly static in tested runs | fit every active hypothesis, generate hypothesis-specific probes, compile evidence, and revise only when a later numeric round can consume the action | a recorded revision is meaningful only if it can affect subsequent training/querying |
| evidence | post-fit metrics and small holdouts were easy to mix with fitted data | predictions are frozen before Oracle answers; deterministic warmup pair families are split wholly into train, prequential-selection validation, and final-calibration roles | avoid training-label rescoring, correlated-family leakage, and threshold-selection leakage |
| Oracle allocation | query selection was weak and Oracle cost tended to grow with the number of hypothesis-specific probes | a query-priority beam and pending interventions form one shared candidate pool; sequential global acquisition spends one bounded per-round budget and shares every answer with the whole bank | reduce Oracle cost and compare hypotheses on common evidence |
| label-balance acquisition | one soft reserved safe slot, selected as a batch, could keep failing without retry | trainable safe/violation deficits are recomputed after every answer; all balance-seeking shares a two-slot cap, while the safe side additionally requires causal source crossings and clause-aware rejection-signature deduplication | improve label coverage without allowing quota filling to consume the whole information budget |
| falsifier | heuristic smooth objective, fixed `0.08` radius, and the final optimizer iterate could be queried even without a real crossing | model-safe anchors, calibrated generation margin `threshold+0.05`, kinematically valid checkpoints, scan-bracket refinement to query margin `threshold+0.02`, and a queryability gate | the requested model/clause must actually cross its decision boundary before Oracle budget is spent |
| falsifier scale | one small radius or an increasingly expensive ladder | the five-seed comparison selected a single `0.32` radius; `0.04/0.08/0.16/0.32` remains an explicit ablation | the ladder did not improve Oracle-safe yield or query deformation, while using about `3.26x` optimizer launches |
| candidate-trajectory support | warmup and falsifier queries were deformations of a selected expert path | this remains expert-anchored: warmup scales the demonstrated detour, while outer-loop search starts from an expert/chord interpolation plus smooth basis restarts; only false-unsafe has the hard `0.32` pointwise trust radius | certification and acquisition improved, but trajectory-space coverage did not become global |
| numeric fitting | per-member bootstrap could omit 25–50% of a small paid query buffer; all states could receive violation MIL credit | every member sees every trainable query; source-matched violations exclude selected-feature states unchanged from their known-safe anchor | stabilize fitting and prevent a shared safe endpoint/start state from becoming a cheap violation witness |
| multiple constraints | a composite could be optimized or refined through the wrong clause; early clauses could monopolize probes | generation, hard certificates, refinement, query gating, deduplication, and cross-round scheduling all carry an explicit clause identity | feature-A equality and feature-B inequality must be tested separately rather than credited through one nuisance clause |
| champion selection | one scalar mixed predictive evidence, intervention yield, uncertainty, complexity, and parameter count | qualification gates run before ranking; nested/progress/dynamics evidence priors require material gain, while terminal representation collision and linear-max convex support provide exact structural contradictions | acquisition statistics and implementation size must not rescue a structure logically contradicted by public evidence, while heuristic priors remain distinguishable from proofs |
| finalization | a selected model could be directly refit | an optional diagnostic-profile finalizer fits candidates without mutating the incumbent, calibrates on `final_calibration`, selects once on disjoint `warmup_validation`, and commits only a candidate that still passes the gates | post-selection training must not silently invalidate the public qualification evidence |
| privacy and reproducibility | a short result summary | capability-limited membership Oracle, evaluation-only metrics computed only after that round's evidence/semantic decision, query/stage/finalization diagnostics, frozen-bank replay, resolved plans, and source/input/runtime hashes | make every label, selection decision, and treatment auditable without feeding private metrics back into learning |

The invariant that survived every revision is: **the LLM proposes semantic
structure; demonstrations and whole-trajectory Oracle answers determine the
numeric boundary**. The learner still receives no obstacle center/radius or
ground-truth violation time. A compact current-round workflow and the complete
change rationale are appended to [ARCHITECTURE.md](ARCHITECTURE.md#7-架构演进记录2026-08-18保留原设计).

## Legacy high-level data-flow sketch (historical)

```text
Task text + variable schema
          │
          ▼
 GPT/Qwen: generate candidate hypothesis population H1...HK
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
 GPT/Qwen: retain / retire / replace / split / intervention
          └─────────────────────────────── next outer round
```

The final obstacle boundary is still learned by the neural network. What the LLM decides is the constraint search space and the follow-up experiments; it never outputs centers, radii, thresholds, or boundary points.

This retained sketch shows the semantic–numeric idea, not the exact current
query order. In the current loop, candidates first enter one shared acquisition
pool and only a bounded subset reaches the Oracle; see
[ARCHITECTURE.md §7.1](ARCHITECTURE.md#71-三个架构阶段). The command-line default
still selects a Qwen profile. The latest five-seed mechanism studies instead
used a diagnostic configuration, a frozen GPT round-0 bank,
`freeze_revisions=true`, and zero live LLM calls; they validate fitting,
falsification, acquisition, and structural gates, not live multi-round semantic
reasoning. Optional finalization is enabled only by profiles that request it.

## Implemented capabilities

- A hypothesis population and version space, rather than a single irrevocable hypothesis;
- Variables: `x_position/y_position/x_velocity/y_velocity/speed/progress`;
- Atomic relations: `forbidden_region`, `upper_bound`, `lower_bound`, and `equality_band`;
- Multi-constraint hypotheses: safe iff every clause is satisfied, violation iff any clause is violated;
- Both joint and independent coupling;
- `forbidden_region` supports an implicit MLP as well as a linear neural control head;
- Univariate `upper_bound/lower_bound` uses learnable threshold heads;
- `max/mean/last` trajectory aggregation, where the choice changes the computation graph;
- Trajectory-level multiple-instance learning with latent witnesses;
- Numeric fitting uses every trainable query for every ensemble member; the
  disjoint warmup-validation and final-calibration records remain excluded
  from gradients. For a
  source-matched Oracle violation derived from a known-safe expert, max-MIL
  excludes selected-feature states unchanged from that anchor; this prevents
  shared safe endpoints from becoming spurious violation witnesses;
- Ensemble epistemic uncertainty;
- Differentiable, feature-generic falsification with smooth basis restarts;
- Cross-hypothesis query acquisition using disagreement, boundary proximity, uncertainty, novelty, and source potential;
- Sequential safe-query acquisition: frozen candidate predictions, trainable-label deficits, Oracle-feedback retries, a two-slot balance cap, and rejection-signature deduplication;
- Safe probes require a certified source-model crossing at both the generation endpoint and the refined query point; all-model-safe coverage candidates cannot consume the safe quota;
- False-unsafe synthesis uses calibrated hard margins, model-safe expert anchors, and kinematically valid iterate checkpoints; the default search is the 5-seed-selected single `0.32` trust radius, while a multi-scale ladder remains an explicit ablation option;
- For composite hypotheses, the full model and the requested clause must both cross their margins, so an equality clause cannot be credited for an unrelated inequality-clause rejection;
- Failed or uncontrollable false-unsafe searches remain diagnostics and cannot consume Oracle budget; successful endpoints are scan-bracketed and refined while retaining a separately recorded query margin;
- Pure-terminal hypotheses are structurally rejected when identical terminal representations carry both safe and violation labels;
- Atomic 1D/2D joint linear-max `forbidden_region` hypotheses have a leakage-free convex-support-order gate. The containment test is strictly one-sided, requires two distinct safe anchors, and can be switched to audit-only without changing scores or query priorities;
- Prequential/held-out structure evidence rather than post-fit training accuracy;
- Qwen or GPT reads structured evidence each round and returns typed revision actions;
- GPT uses Responses API Structured Outputs with a strict JSON Schema; local Qwen uses tolerant surface repair followed by the same trusted compiler;
- Compilation failures, illegal variables, illegal relations, and contradictory actions are logged and rejected before application when validation catches them; an unexpected failure during action application stops the run rather than claiming a general transactional rollback;
- When Qwen output is truncated, only JSON objects that are already closed and pass compilation are recovered and accepted; malformed objects are never executed;
- Successful GPT/Qwen responses, parse repairs/errors, fallbacks, hypothesis versions, and all Oracle queries are auditable; backend failures default to fail-fast unless fallback is explicitly enabled;
- The hidden circular ground truth is accessed only by the experiment harness's membership Oracle and evaluation-only code. Per-round evaluation happens after that round's semantic action is produced and is never fed into evidence, prompts, acquisition, or champion selection;
- Ablation scripts for the closed loop and for a frozen one-shot bank.

## Project layout

```text
llm_modulo_cegis/
|-- run_obstacle_avoid.py       # shared training CLI and Obstacle2D entry point
|-- run_semtraj2d.py            # SemTraj2D training preset
|-- run_carrywater_active.py    # CarryWaterActive training preset
|-- src/llm_modulo_cegis/       # reusable implementation; no experiment orchestration
|   |-- data.py                 # trajectory I/O, schemas, and derived features
|   |-- hypotheses.py           # typed constraint IR, compiler, bank, and revisions
|   |-- semantic.py             # GPT/Qwen/frozen/deterministic semantic reasoners
|   |-- learner.py              # multi-hypothesis fitting and trajectory-level MIL
|   |-- falsifier.py            # optimization-based counterexample generation
|   |-- pool_falsifier.py       # direct candidates from task-specific trajectory pools
|   |-- evidence.py             # leakage-free numeric evidence for semantic revision
|   |-- oracle.py               # budgeted whole-trajectory binary Oracle interface
|   |-- loop.py                 # bi-level semantic/numeric CEGIS controller
|   |-- evaluation.py           # evaluation-only ground-truth metrics and plots
|   |-- structure_evaluation.py # post-fit structural diagnostics
|   |-- carrywater_active.py    # CarryWaterActive task adapter
|   `-- types.py                # shared dataclasses and result records
|-- configs/                    # task configs, hypothesis banks, and multi-seed plans
|   |-- obstacle_avoid_*.yaml
|   |-- semtraj2d_*.yaml
|   |-- carrywater_active_*.*
|   `-- *_multiseed_plan.yaml
|-- data/                       # versioned benchmark inputs
|   |-- Obstacle2D/
|   |-- SemTraj2D/
|   `-- CarryWaterActive/
|-- tools/                      # offline generation, private evaluation, and replay
|   |-- generate_semtraj2d.py
|   |-- generate_carrywater_active.py
|   |-- evaluate_semtraj2d.py
|   |-- evaluate_carrywater_active.py
|   |-- replay_finalization.py
|   `-- README.md
|-- experiments/                # gated multi-seed studies and historical ablations
|   |-- run_carrywater_q48_multiseed.py
|   |-- run_ablation.py
|   |-- run_falsifier_multiseed.py
|   |-- run_numeric_fitting_multiseed.py
|   |-- run_violation_pooling_multiseed.py
|   |-- run_linear_max_support_gate_multiseed.py
|   `-- README.md
|-- tests/                      # unit, integration, benchmark, and layout checks
|-- outputs/                    # generated runs and immutable historical snapshots
|-- reports/                    # experiment conclusions and implementation records
|-- ARCHITECTURE.md             # detailed design and change history
|-- llm_modulo_cegis_architecture.pdf
|-- pyproject.toml
`-- requirements.txt
```

The root intentionally exposes only the three normal training commands. Runtime
code flows from those entry points into `src/llm_modulo_cegis/`. The scripts in
[`tools/`](tools/README.md) generate or evaluate artifacts offline and are not
part of model selection. The scripts in [`experiments/`](experiments/README.md)
orchestrate reproducible sweeps around the same training entry points; learning
logic must remain in `src/llm_modulo_cegis/` rather than being duplicated there.
`outputs/` and `reports/` contain artifacts, not importable runtime code.

Reports dated on or before 2026-08-21 may show the former flat paths. Translate
`llm_modulo_cegis/run_*multiseed.py` and `run_ablation.py` to
`llm_modulo_cegis/experiments/<same-name>`, and translate root-level
`generate_*.py`, `evaluate_*.py`, and `replay_finalization.py` to
`llm_modulo_cegis/tools/<same-name>`. The three primary `run_*.py` paths shown
above are unchanged. Existing outputs and their historical implementation
manifests were deliberately not relocated or rewritten.

## Running

The project's designated `pucl` environment can run the non-LLM closed loop and the tests:

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe -m unittest discover -s llm_modulo_cegis\tests -v

C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_smoke.yaml
```

GPT (the project-root `api.env` is loaded without logging the key):

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_gpt_smoke.yaml
```

To test only GPT hypothesis generation, without loading demonstrations, private
evaluation data, or the Oracle, add `--semantic-only`:

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_gpt_smoke.yaml `
  --semantic-only
```

The GPT backend does not silently convert API/network failures into a successful fallback run. Set `semantic.fallback_on_backend_error: true` only for an explicitly desired offline-degradation experiment. Semantic compilation failures can still use `allow_fallback`, and every such event is audited.

Revision calls are made only when another numeric round remains. Thus every
recorded retire/replace/add/intervention action can affect a subsequent round,
and the terminal round does not spend an LLM call on actions that cannot be
trained or evaluated.

For a slower diagnostic run with balanced expert auditing, champion qualification
gates, a rolling two-round prequential window, and a larger query/training budget:

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_gpt_diagnostic.yaml
```

For a controlled acquisition ablation, replay a saved round-0 hypothesis bank.
This makes no LLM call, freezes semantic revisions, records the source artifact
hash, and keeps both arms on the same initial structures:

```powershell
# Proposed safe-query policy
C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_gpt_diagnostic.yaml `
  --initial-hypothesis-bank llm_modulo_cegis\outputs\gpt_safe_acquisition_final\hypothesis_bank.json `
  --output llm_modulo_cegis\outputs\frozen_bank_safe_acquisition

# Same initial bank and generator configuration, global acquisition only.
# Later candidate pools may diverge after the two policies observe different labels.
C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_gpt_diagnostic.yaml `
  --initial-hypothesis-bank llm_modulo_cegis\outputs\gpt_safe_acquisition_final\hypothesis_bank.json `
  --global-acquisition-only `
  --output llm_modulo_cegis\outputs\frozen_bank_global_acquisition
```

The implementation rationale, failure analysis, and fixed-bank A/B results are
documented in `reports/SAFE_QUERY_ACQUISITION_2026-08-17.md`.
The hard-margin/radius-ladder design and fixed-bank 2x2 falsifier ablation are
documented in `reports/FALSIFIER_HARD_MARGIN_LADDER_2026-08-17.md`.
The follow-up five-seed paired experiment is documented in
`reports/FALSIFIER_MULTISEED_SINGLE_VS_LADDER_2026-08-17.md`. It selected the
cheaper single `0.32` radius as the default. Use
`--false-unsafe-radius-ladder 0.04 0.08 0.16 0.32` only for an explicit ladder
experiment; the sweep runner passes both arm settings explicitly.

The numeric-fitting follow-up is documented in
`reports/NUMERIC_FITTING_STABILITY_2026-08-17.md`. Its strict three-arm,
five-seed experiment selected full-trainable-buffer fitting plus source-anchor
changed-state MIL as the default: spatial was qualified in `5/5` seeds and was
the final champion in `4/5`, versus `2/5` champions under classic bootstrap.
The remaining seed selected a simultaneously qualified affine competitor, so
the residual issue was structure ranking rather than numeric qualification.
The follow-up [linear-max support-gate report](reports/LINEAR_MAX_SUPPORT_GATE_2026-08-17.md)
replaces penalty tuning with an exact public structural certificate. A frozen
five-seed audit-only/enforced comparison produced qualified spatial champions
in `5/5` seeds in both arms. The enforced gate is therefore adopted as a sound
guard with no observed public-floor regression, but the tied result is not a
performance-improvement claim.

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\experiments\run_falsifier_multiseed.py `
  --plan llm_modulo_cegis\configs\falsifier_multiseed_plan.yaml `
  --output-root llm_modulo_cegis\outputs\falsifier_multiseed_5seed_new

C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\experiments\run_violation_pooling_multiseed.py `
  --plan llm_modulo_cegis\configs\violation_pooling_multiseed_plan.yaml `
  --output-root llm_modulo_cegis\outputs\violation_pooling_multiseed_5seed_new

C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\experiments\run_linear_max_support_gate_multiseed.py `
  --plan llm_modulo_cegis\configs\linear_max_support_gate_multiseed_plan.yaml `
  --output-root llm_modulo_cegis\outputs\linear_max_support_gate_multiseed_5seed_rerun `
  --max-workers 3
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
C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\experiments\run_ablation.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_smoke.yaml
```

## Outputs and auditability

Every run produces:

- `semantic_interactions.json`: semantic-backend prompts/responses, parse errors, and final actions when a backend response is available;
- `hypothesis_bank.json`: the admission, retention, and retirement history of each hypothesis;
- `evidence_history.json`: the leakage-free evidence actually fed back to the LLM;
- `evaluation_history.json`: ground-truth evaluation kept in isolation, never fed back to the LLM;
- `oracle_queries.npz` and `oracle_query_log.json`: whole-trajectory queries;
- `query_diagnostics.json`: every generated candidate (queried or not), actual query sequence, label deficit, estimated safe probability, causal rejectors, boundary-refinement status, trajectory ranges, acquisition components, Oracle label, and every hypothesis's frozen pre-query prediction;
- `stage_diagnostics.json`: expert split summaries plus pre-query/post-query model snapshots, learned scalar thresholds in raw units, and fit/audit/evaluation-only expert predictions;
- `finalization_diagnostics.json`: all-expert refit, disjoint calibration, commit/rejection reasons, and incumbent-versus-candidate metrics;
- `constraint_models.pt`: all hypothesis models plus the final champion;
- `learned_boundary.png`: the final neural boundary;
- `semantic_trace.png`: hypothesis scores and semantic actions;
- `result.json`: the final summary.

## Current limitations

1. `ObstacleAvoid` has only two-dimensional observations, so the evidence for structure identification is weak; a formal paper must add distractor dimensions and heterogeneous constraints.
2. The oracle returns whole-trajectory labels only. Source-anchor masking can
   exclude causally unchanged states, but neither that mask nor the remaining
   MIL latent witness is a ground-truth violation-time label.
3. Qwen2.5-1.5B still has limited semantic reasoning. Surface-format omissions and common enum aliases are repaired, but unknown variables or illegal semantics remain rejected. GPT is the recommended reasoning backend when external API use is permitted.
4. The smoke configuration only verifies the closed loop; it says nothing about boundary quality. Use the full training schedule for evaluation.
5. The current model family implements MLP, linear, and scalar threshold heads. `GP/SDF/symbolic formulas` can be added through the compiler interface, but are not pretended to be supported already.
6. The convex-support gate is deliberately incomplete: it covers only atomic 1D/2D joint linear-max forbidden-region hypotheses. It does not settle general ranking among multiple structurally viable qualified models.
7. Every current Oracle-query candidate is expert-anchored. This gives a plausible local counterfactual and, for false-unsafe, a known-safe reference; it also biases coverage toward demonstrated path families. Smooth basis restarts may reach different regions, but they do not guarantee a new homotopy class.
8. A direct spline generator and a local/global hybrid are proposed next steps only. No such generator, controlled comparison, or performance result exists in the current code or artifacts. Here, **global acquisition** means global selection from one shared cross-hypothesis candidate pool; it does not mean global trajectory synthesis.

## SemTraj2D publication benchmark (added 2026-08-21)

The original Obstacle2D data remain useful as a closed-loop smoke test, but they
cannot separate proxy-variable failures, simple scalar constraints, dynamic
constraints, or heterogeneous compositions.  A new deterministic benchmark is
therefore available at [data/SemTraj2D](data/SemTraj2D/README.md).

SemTraj2D adds 12 differentiable candidate features and eight tasks: a paired
clean/confounded obstacle experiment, a linear halfspace, a scalar equality
band, a direction-balanced speed upper bound, spatial-plus-speed and
equality-plus-inequality composites, and an open-set eventual-visit negative
control.  Public safe demonstrations are separated from
analytic private rules and independently generated private trajectory probes.
In particular, `disk_clean` and `disk_upper_proxy` use exactly the same hidden
rule and private evaluation bank; only the expert route distribution changes.

Two authorized GPT semantic-only checks produced schema-valid hypotheses
without fallback: `disk_upper_proxy` accepted three candidates whose first is
the intended joint `(x,y)` keep-out structure, while `lane_and_speed` accepted
two candidates whose first contains the intended `y` equality band and speed
upper-bound clauses.  A subsequent two-round, 26-query closed-loop smoke froze
the correct typed disk structure but remained `inconclusive`: private trajectory
balanced accuracy was `0.641`, safe accuracy `0.281`, violation recall `1.000`,
and worst-group balanced accuracy `0.500`.  The result localizes the present
failure to conservative numeric fitting rather than LLM schema fallback.  It
is evidence that the benchmark exposes the unseen-route weakness; it is not an
architecture-superiority claim.  See the
[benchmark implementation report](reports/SEMTRAJ2D_BENCHMARK_2026-08-21.md)
and the fixed multi-seed, equal-budget protocol in the dataset README.

## Information and Oracle assumption

The learner receives three user-facing channels: a task description, safe demonstrations, and optional binary feasibility judgments for selected whole trajectories. It never receives the hidden obstacle geometry or violation time. This is a membership-query active-learning setting, not demonstration-only inverse constraint learning.

Oracle use is bounded and auditable. Whole warmup pair families are assigned to gradient training, `warmup_validation` selection evidence, or `final_calibration`; a separate set of known-safe experts is reserved for structure-audit safety. In each outer round, the active query beam first generates candidates and freezes every model prediction without Oracle access. The shared budget is then spent sequentially: after each answer, only the observed trainable-label deficit, empirical intervention yield, and confirmed causal-safe signature may change; models are not retrained until the batch ends. Every selected answer is shared by every active hypothesis through its frozen pre-query prediction. Report both total Oracle calls and performance-versus-query curves, and include a frozen-bank acquisition ablation when evaluating whether safe-label balancing justifies displaced global queries.

For the detailed mathematics and implementation boundaries see [ARCHITECTURE.md](ARCHITECTURE.md) and the rebuilt [architecture PDF](llm_modulo_cegis_architecture.pdf); for baseline measurement records see [reports/IMPLEMENTATION_AND_TEST.md](reports/IMPLEMENTATION_AND_TEST.md); for the Qwen closed-loop test on ObstacleAvoid see [reports/OBSTACLE_AVOID_QWEN_TEST.md](reports/OBSTACLE_AVOID_QWEN_TEST.md); for the authorized GPT smoke and its limitations see [reports/GPT_SMOKE_2026-08-17.md](reports/GPT_SMOKE_2026-08-17.md); for the gated diagnostic rerun see [reports/GPT_GATED_DIAGNOSTIC_2026-08-17.md](reports/GPT_GATED_DIAGNOSTIC_2026-08-17.md); for the proxy-variable guard and transactional finalization see [reports/GPT_PROXY_GUARDED_IMPLEMENTATION_2026-08-17.md](reports/GPT_PROXY_GUARDED_IMPLEMENTATION_2026-08-17.md); for the intervention-witness architecture and final integrated GPT result see [reports/GPT_INTEGRATED_FINAL_2026-08-17.md](reports/GPT_INTEGRATED_FINAL_2026-08-17.md); for the exact affine-max structural rejection and its paired audit see [reports/LINEAR_MAX_SUPPORT_GATE_2026-08-17.md](reports/LINEAR_MAX_SUPPORT_GATE_2026-08-17.md).

## CarryWaterActive benchmark (added 2026-08-21)

CarryWaterActive complements SemTraj2D with a 12-dimensional, dynamically
replayable manipulation benchmark.  Its intended constraint is a heterogeneous
conjunction: track the requested carrying height, bound direction-independent
three-dimensional speed, and keep the cup's total tilt small while leaving yaw
free.  The learner sees 40 fit experts, 12 validation experts, 12
structure-audit experts, public dynamics, and a 512-trajectory **unlabelled**
candidate pool.  Those candidates are sampled as new control-space rollouts;
they do not perturb or reuse expert waypoints.

The sibling private bundle is never mounted as learner input.  It contains 512
matched safe/unsafe pairs (1,024 trajectories) covering height-only,
speed-only, tilt-only, and multi-clause violations.  Exact representation
collisions make the three tempting shortcuts falsifiable: opposite-labelled
pairs share the full world-`z` sequence, one velocity component, or yaw,
respectively.  Report trajectory balanced accuracy together with worst pair
target balanced accuracy, exact-pair accuracy, pair-ranking accuracy, and
minimum clause recall; aggregate accuracy alone is insufficient.

The frozen 13-query connectivity smoke completed without loading private
evaluation, but correctly remained `inconclusive`.  Its provisional world-`z`
proxy obtained post-hoc balanced accuracy `0.500`, worst pair-target balanced
accuracy `0.500`, and exact-pair accuracy `0.000`.  A diagnostic evaluation of
all seven frozen-bank learners also showed that this tiny query budget did not
fit the intended composite.  These are useful failure-localization results,
not a performance claim.  In a separate authorized GPT semantic-only run,
GPT accepted four schema-valid hypotheses with zero fallback; the first was the
intended height-band + speed-bound + tilt-bound composite and used zero Oracle
queries.

```powershell
# Cheap end-to-end connectivity check (frozen hypothesis bank).
C:\Users\rpc21\miniconda3\envs\pucl\python.exe `
  llm_modulo_cegis\run_carrywater_active.py `
  --output outputs\carrywater_active_smoke

# Semantic generation only; api.env is loaded automatically.
C:\Users\rpc21\miniconda3\envs\pucl\python.exe `
  llm_modulo_cegis\run_carrywater_active.py `
  --config llm_modulo_cegis\configs\carrywater_active_gpt_semantic.yaml `
  --semantic-only `
  --output outputs\carrywater_active_gpt_semantic

# Private evaluation is a separate post-freeze command.
C:\Users\rpc21\miniconda3\envs\pucl\python.exe `
  llm_modulo_cegis\tools\evaluate_carrywater_active.py `
  --run-dir outputs\carrywater_active_smoke
```

The preceding expert-anchored limitation remains historically accurate for
the Obstacle2D falsifier.  CarryWaterActive adds a task-specific global-rollout
pool adapter; it does not retroactively turn the old spline proposal into a
generic generator.  See the [dataset note](data/CarryWaterActive/README.md) and
the [implementation and evaluation report](reports/CARRYWATER_ACTIVE_BENCHMARK_2026-08-21.md).

### Q48 gated pilot (seed 7, development-only)

The preregistered five-seed comparison was not launched blindly.  A seed-7
pilot first compared the oracle-structure numeric ceiling (the correct
three-clause composite alone) with the frozen seven-hypothesis bank at the same
48-query budget.  Both arms recovered the exact composite as a publicly
qualified champion, but neither passed the private clause-balanced release
gate.  The correct-only arm obtained trajectory balanced accuracy `0.7813`,
exact-pair accuracy `0.5625`, worst pair-target balanced accuracy `0.5000`, and
minimum clause recall `0.2857`.  The full-bank arm obtained `0.7461`, `0.4922`,
`0.5000`, and `0.2857`, respectively.  Consequently the formal five-seed run
and matched PUCL/all-feature-MLP baselines remain gated off.

A post-freeze development audit localized the failure.  The public candidate
pool contains exactly 128 safe, 128 height-only, 128 speed-only, and 128
tilt-only rollouts, so the generator is not missing height evidence.  In the
correct-only model each individual clause head orders its corresponding
violations correctly, but the raw head offsets differ.  Applying one global
threshold after a raw cross-clause `max` therefore misses all height-only
rollouts while preserving expert safety.  The fixed four-trajectory threshold
calibration split also happened to contain tilt, safe, speed, and tilt, with no
height example.  The next architectural experiment is thus clause-specific
thresholds (or calibrated per-clause margins before `any_violation/max`) with
disjoint, clause-targeted calibration evidence.  The full-bank arm additionally
needs a fixed per-clause acquisition reservation; increasing the total query
budget or regenerating the existing single-clause pool is not the first remedy.

The Q48 runner enforces the exact `12 + 3 x 12 = 48` Oracle budget, advances
through the full public pool rather than recycling the first 64 candidates,
fails on query underfill, restores the best publicly qualified checkpoint, and
seals the model, query ledger, public evidence, threshold history, checkpoint
history, and stage diagnostics before private evaluation.  The current test
suite passes `133/133` tests.  The pilot artifacts are under
`outputs/q48_pilot_checkpoint_retention_correct_seed7/` and
`outputs/q48_pilot_checkpoint_retention_full_seed7/`.
