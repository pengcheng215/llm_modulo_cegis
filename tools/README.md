# Offline tools

These commands are not part of the CEGIS training loop:

- `generate_semtraj2d.py` and `generate_carrywater_active.py` build benchmark
  bundles.
- `evaluate_semtraj2d.py` is the generic post-freeze private evaluator.
- `evaluate_carrywater_active.py` supplies the CarryWaterActive private path to
  that evaluator.
- `replay_finalization.py` replays final model selection without GPT or Oracle
  calls.

Run them from the workspace root, for example:

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe `
  llm_modulo_cegis\tools\evaluate_carrywater_active.py `
  --run-dir outputs\carrywater_active_smoke
```
