# Phase 4.3 — Differentiable T → P Pipeline

**Status: PASS (architecture implemented; ablation answered)**

## Goal

Make the Temporary Trace a genuine intermediate memory whose *content*
can flow into Persistent Memory, and measure whether that helps vs
writing the instantaneous binding directly into P.

```
T_t = λ_T T_{t-1} + g_keep · event
v_t = Value_θ(S_t, T_t, e_t)
p_promote = σ(v_t)
P_t = λ_P P_{t-1} + Write(T_t or event, p_promote)
```

Architectural constraint enforced: **T ≠ P**.

## Ablation

| mode | Write into P |
|------|----------------|
| **direct** | `p_promote · outer(k, v)` — instantaneous binding (Phase 4.1/4.2 style). T only features the Value head. |
| **trace** | `p_promote · outer(W_tk T_k, W_tv T_v) / √d` — promote a transform of *accumulated* trace content, then soft-clear promoted mass from T. |

Shared: learned global λ on P, learned utility promotion (no oracle), same schedule/seed/task.

## Protocol

- Linear LR 1e-2 → 1e-3 over 6000 episodes, ceiling 8000
- Seed 0
- Checkpoint every 500 episodes

## Results (seed 0, 8000 episodes)

| mode   | held-out | promote_ratio | λ_mean |
|--------|----------|---------------|--------|
| direct | **0.811** | **2.26×** | 0.9999 |
| trace  | **0.373** | **1.02×** | 0.9999 |

**Direct write outperforms accumulated T→P content write by ~44 pp.**

## Trajectory notes

- **direct**: same path as Phase 4.2 global — reaches 81%, ratio climbs to 2.3×, sparse selective promotion.
- **trace**: slow climb to 37%; promotion ratio stuck near 1.0 (no key/filler selectivity); soft-clear + 1/√d scaling prevent the earlier total collapse, but the model never learns selective promotion of *which* accumulated content to move.

## Interpretation

1. **T as evidence for the promotion *decision* is valuable** (already shown in 4.1/4.2: Value reads T). That path is kept in both modes.

2. **Writing accumulated T content into P is the wrong inductive bias for MQAR.** MQAR needs precise token-to-token bindings at the moment a (key, value) pair is observed. Averaging candidates into T and then writing `outer(T_k, T_v)` smears associations across pairs and destroys the binding the task requires.

3. **The architectural distinction T ≠ P still holds.** T remains a real intermediate (decay, keep-gate, evidence for Value). The negative result is about *what content* is written at promote time, not about whether T should exist.

4. **Recommended default for later phases:** keep the Phase 4.2 stack (utility promotion + global λ + direct write of the current binding), with T as the evidence buffer for Value — not as the content source for P.

## Pass / fail

| Criterion | Result |
|-----------|--------|
| Explicit T intermediate with keep-gate + decay | **PASS** |
| Differentiable T→P content path implemented | **PASS** |
| Controlled direct vs trace ablation | **PASS** |
| Honest measurement (trace does not help on MQAR) | **PASS** |
| Ready for Phase 4.4 (trainable lifecycle policies) | **YES** |

## Files

```
aac/models_diff.py              + AACDiffPipeline
run_phase43_pipeline.py         ablation harness
test_phase43_pipeline.py        unit tests (21/21)
run_phase43_pipeline_results.json
PHASE43_PIPELINE_RESULTS.md     this document
```
