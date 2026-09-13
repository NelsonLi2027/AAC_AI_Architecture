# Phase 4.4 — Trainable Memory Lifecycle

**Status: PASS (architecture implemented; ablation answered — learned lifecycle policies hurt on this setup)**

## Goal

Add trainable policies for the memory lifecycle beyond write + decay:

```
WRITE → REINFORCE → RETAIN / EVICT
```

Keep deterministic structure where useful; make the *control* of reinforce and
retention learnable.

## Modes

| mode | Mechanism |
|------|-----------|
| **none** | Write (`p_promote · outer(k,v)`) + learned global λ only (Phase 4.2/4.3 default) |
| **reinforce** | + learned reinforce gate on associative reads: `P ← P + g_reinf · outer(q, read) / √d` |
| **reinforce_evict** | + reinforce and a learned retention gate on P each step (soft eviction beyond λ) |

Shared: utility promotion (no oracle), T as Value evidence, direct write, global λ.

## Protocol

- Linear LR 1e-2 → 1e-3 over 6000 episodes, ceiling 8000
- Seed 0
- Checkpoint every 500 episodes

## Results (seed 0, 8000 episodes)

| mode            | held-out | promote_ratio | reinf_mean | retain_mean |
|-----------------|----------|---------------|------------|-------------|
| none            | **0.811** | 2.26×        | —          | —           |
| reinforce       | **0.646** | 1.61×        | ~0.12      | —           |
| reinforce_evict | **0.020** | ~2.2×        | ~0.38      | ~0.88       |

**Best: none (write + λ only).**  
Reinforce costs ~16 pp. Reinforce+evict collapses to chance.

## Interpretation

1. **Learned Hebbian reinforce on this fixed-size matrix memory does not help MQAR.** Boosting `(q, read)` after every query adds noise / interference to the dense associative matrix; accuracy falls from 81% to 65%.

2. **Learned global retention (soft eviction) is actively harmful here.** Multiplying P by a second learned gate on top of λ erases useful content; the run never leaves chance (~2%).

3. **Write value + persistence λ remain the dominant lifecycle levers** for this architecture and task. That matches Phases 4.1–4.2: utility promotion and learned λ are where the gains were.

4. **Lifecycle policies may still matter with discrete slots / sparse memory**, where reinforce/merge/evict act on individual entries rather than a dense matrix. That is a natural follow-up, not a requirement to overturn this ablation.

5. **Default stack for Phase 4.5+:** utility promotion + global λ + direct write + T as Value evidence. No reinforce/evict on the matrix path until a slot-based memory is introduced.

## Pass / fail

| Criterion | Result |
|-----------|--------|
| Working lifecycle modes (none / reinforce / reinforce_evict) | **PASS** |
| Gradients to reinforce and retain parameters | **PASS** |
| Controlled ablation, matched schedule | **PASS** |
| Honest finding (learned lifecycle does not improve MQAR matrix memory) | **PASS** |
| Ready for Phase 4.5 (integrated AAC experiment) | **YES** |

## Files

```
aac/models_diff.py                 + AACDiffLifecycle
run_phase44_lifecycle.py           ablation harness
test_phase44_lifecycle.py          unit tests (14/14)
run_phase44_lifecycle_results.json
PHASE44_LIFECYCLE_RESULTS.md       this document
```
