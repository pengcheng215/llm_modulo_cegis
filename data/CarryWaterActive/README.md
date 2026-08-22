# CarryWaterActive

CarryWaterActive is a synthetic active constraint-identification benchmark. The learner receives only `public/carrywater_active/`, including 64 known-safe experts and a 512-trajectory unlabeled candidate pool. `splits.json` uses `train/validation/test = 40/12/12`; `test` is reserved as the structure-audit expert split and is not a private safety test.

Every observation is replayable under the public dynamics in `public/carrywater_active/dynamics_spec.json`. Actions have shape `[T-1, 6]`, so there is no fabricated terminal action. The sibling `private/carrywater_active/` directory backs the capability-limited membership Oracle and post-hoc evaluator and must not be mounted as learner input. The candidate archive deliberately contains no `labels`, clause annotations, thresholds, or private seed.
