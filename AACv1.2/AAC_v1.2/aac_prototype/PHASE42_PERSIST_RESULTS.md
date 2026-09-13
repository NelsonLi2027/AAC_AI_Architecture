# Phase 4.2 — Learned / State-Dependent Persistence

**Status: PASS — learned persistence improves over fixed λ**

## Goal

Replace fixed persistence `λ = 0.999` with learnable forms and measure
whether adaptive retention improves MQAR accuracy under the same
utility-based promotion path from Phase 4.1.

```
λ_t = sigmoid(f_θ(...))
P_t = λ_t · P_{t-1} + p_promote · outer(k, v)
```

## Modes (controlled ablation)

| mode   | λ definition |
|--------|----------------|
| fixed  | constant 0.999 (Phase 4.1 baseline) |
| global | single learnable scalar `λ = sigmoid(b_λ)` |
| state  | per-step `λ_t = sigmoid(MLP(S_t, T_t, emb))` |

All modes share:
- learned Value → promote (no oracle training target)
- fixed temporary-trace decay `decay_t = 0.9`
- identical LR schedule, task, seed, parameter budget (aside from the
  small persistence head)

## Protocol

- Schedule: linear 1e-2 → 1e-3 over 6000 episodes, ceiling 8000
- Checkpoint every 500 episodes
- Seed 0 completed for all three modes (seed 1 partially run)
- Diagnostics: held-out acc, λ_mean, promote ratio, loss

## Results (seed 0, 8000 episodes)

| mode   | held-out | λ_mean | promote_ratio |
|--------|----------|--------|---------------|
| fixed  | **0.676** | 0.999  | 1.66× |
| global | **0.811** | 0.9999 | 2.26× |
| state  | **0.786** | ~1.000 | 2.26× |

**Learned global λ gains +13.5 pp over fixed.**  
State-dependent λ gains +11.0 pp over fixed, slightly behind global on this seed.

## Trajectory notes (seed 0)

- **fixed**: steady climb to 67.6%; λ locked; ratio ~1.5–1.7×
- **global**: tracks fixed early, then pulls ahead after ~4k; λ drifts from
  init ~0.999 toward 1.0; ratio climbs to 2.3×; ends at 81.1%
- **state**: λ saturates near 1.0 by ~1.5k; accuracy ends at 78.6%, between
  fixed and global; ratio also reaches ~2.3×

## Interpretation

1. **Adaptive persistence is valuable.** Both learned modes beat fixed λ
   by double-digit percentage points under matched conditions.

2. **Global λ is sufficient (and slightly better) on this task.** A single
   learnable persistence scalar captures most of the gain. State-dependent
   λ_t did not outperform global here — it quickly saturated to ~1.0,
   behaving like a near-hard retention policy.

3. **Promotion selectivity co-moves with better persistence.** Both learned
   modes reach higher promote ratios (~2.3× vs ~1.7×), suggesting the
   Value estimator and the retention policy reinforce each other when
   memory is allowed to keep what is written.

4. **Still below oracle (~100%) and the long-run utility ceiling (~86%
   under fixed λ at 12k).** Longer training and Phase 4.3 (richer T→P)
   remain open levers.

## Pass / fail

| Criterion | Result |
|-----------|--------|
| Working fixed / global / state implementations | **PASS** |
| Gradients flow to λ parameters | **PASS** |
| Controlled ablation, matched schedule/seeds | **PASS** |
| Learned persistence beats fixed λ | **PASS** (+11–14 pp) |
| Diagnostics (λ, promote ratio) logged | **PASS** |
| Ready for Phase 4.3 (differentiable T→P pipeline) | **YES** |

## Files

```
aac/models_diff.py           + AACDiffPersist
run_phase42_persist.py       experiment harness
test_phase42_persist.py      unit tests (31/31)
run_phase42_persist_results.json
PHASE42_PERSIST_RESULTS.md   this document
```
