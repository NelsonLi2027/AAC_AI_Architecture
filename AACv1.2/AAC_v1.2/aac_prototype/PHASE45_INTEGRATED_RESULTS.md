# Phase 4.5 — Integrated AAC Evaluation

**Status: PASS — end-to-end trainable AAC demonstrated multi-seed**

## Canonical stack (AACDiffFinal)

- Learned utility promotion (Phase 4.1)
- Learned global persistence λ (Phase 4.2)
- Direct write of current binding into P (Phase 4.3)
- No matrix-level reinforce/evict (Phase 4.4)
- T as evidence buffer for Value (T ≠ P)

## Multi-seed results (ceiling 8k)

| config | seed 0 | seed 1 | seed 2 | mean | std |
|--------|--------|--------|--------|------|-----|
| final | 0.811 | 0.751 | 0.627 | **0.730** | 0.077 |
| fixed_lambda | 0.676 | 0.618 | 0.777 | **0.690** | 0.066 |

Final beats fixed-λ on mean (+4.0 pp) and on 2/3 seeds.

## Cross-phase (seed 0)

| Condition | Acc |
|-----------|-----|
| global λ (4.2/4.5) | 0.811 |
| fixed λ (4.2/4.5) | 0.676 |
| T-content write (4.3) | 0.373 |
| + reinforce (4.4) | 0.646 |
| + reinforce+evict (4.4) | 0.020 |
| final 3-seed mean (4.5) | 0.730 ± 0.08 |

## Files

- aac/models_diff.py + AACDiffFinal
- run_phase45_integrated.py
- run_phase45_integrated_results.json
- PHASE45_INTEGRATED_RESULTS.md
