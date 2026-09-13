# Phase 4.1 — Learned Write / Promotion Mechanism

**Status: PASS (mechanism works; utility learns real selectivity without oracle training targets)**

## Goal

Replace oracle / supervised gate labels with a learned estimate of the
future value of information:

```
v_t = Value_θ(e_t, S_{t-1}, T_t)
p_promote = sigmoid(v_t)
```

The only training signal is the downstream MQAR task loss. Oracle write
positions are retained solely for diagnostic logging of promotion rates.

## Implementation

New class `AACDiffUtility` in `aac/models_diff.py`:

| Component | Role |
|-----------|------|
| Fast state `S` | RNN hidden state (unchanged) |
| Temporary Trace `T` | vector evidence accumulator with learned keep-in-trace gate |
| Value estimator | 2-layer MLP on `concat(S, T, emb)` → scalar `v` |
| Promotion gate | `p_promote = sigmoid(v)` soft-writes into persistent matrix `P` |
| Persistent `P` | fixed-decay associative matrix (Phase 4.2 will make decay learned) |

Decision vocabulary supported:
- **IGNORE** — low keep + low promote
- **KEEP IN TRACE** — high keep, low promote
- **PROMOTE** — high promote (evidence flows T → P)

Existing `AACDiffModel` (oracle / supervised / curriculum) is left intact
for controlled ablations.

## Controlled experiment

Script: `run_phase41_utility.py`  
Results: `run_phase41_utility_results.json`

Protocol (identical for all three configs):
- Linear LR schedule 1e-2 → 1e-3 over 3000 episodes
- Ceiling 4000 episodes
- Convergence: 3 consecutive 1000-ep held-out checkpoints within 2 pp
- Seeds 0, 1
- Task: MQAR n_pairs=6, n_queries=6, gap (20,40)

Configs:
| name | mechanism |
|------|-----------|
| oracle | hard ground-truth write positions + decay (upper bound) |
| supervised | learned gate + aux BCE on oracle labels (Phase 3 style) |
| utility | `AACDiffUtility` — no oracle as training target |

## Summary table (held-out accuracy)

| config     | mean  | std   | min   | max   | converged |
|------------|-------|-------|-------|-------|-----------|
| utility    | 0.568 | 0.104 | 0.463 | 0.672 | 0/2*      |
| supervised | 0.279 | 0.245 | 0.034 | 0.524 | 1/2       |
| oracle     | 0.997 | 0.001 | 0.997 | 0.998 | 2/2       |

*Both utility seeds were still climbing at the 4000-ep ceiling; neither
had yet met the strict 3-checkpoint flatness criterion, but both showed
clear upward trajectories and real selectivity.

## Key diagnostics (final checkpoint)

**Utility seed 0** (best):
- promote_key = 0.535, promote_filler = 0.226 → **ratio 2.36×**
- held-out accuracy 67.2 %

**Utility seed 1**:
- promote_key = 0.433, promote_filler = 0.282 → **ratio 1.54×**
- held-out accuracy 46.3 %

**Supervised** (for reference):
- high ratio (1.9–2.9×) but accuracy highly seed-dependent and lower
  under this shorter schedule (consistent with Phase 3 variance).

**Oracle**:
- perfect write timing → ~99.7 % (ceiling on what correct write timing
  alone can achieve with fixed-size memory).

## Interpretation

1. **The learned utility mechanism works.** Without any oracle write
   labels as a training target, `Value_θ` discovers differentiated
   promotion behaviour: important (key) events receive systematically
   higher promotion probability than fillers. Ratio > 1.5× on both seeds.

2. **Task accuracy follows.** Utility reaches 46–67 % under a modest
   4000-ep budget — competitive with or better than the supervised-gate
   baseline on the same schedule, and well above the chance floor.

3. **Oracle remains a strong upper bound** (~100 %). The gap shows there
   is still headroom for better value estimation, longer training, or
   the later architectural pieces (learned persistence, richer T→P
   pipeline, lifecycle policies).

4. **Variance is real.** Seed 0 vs seed 1 for utility (67 % vs 46 %)
   mirrors the seed sensitivity already documented in Phase 3. Multiple
   seeds and trajectory logging remain mandatory.

## Pass / fail decision for Phase 4.1

| Criterion | Result |
|-----------|--------|
| Working implementation of learned Value → promote | **PASS** |
| No oracle labels used as training target | **PASS** |
| Explicit Temporary Trace intermediate | **PASS** (vector T) |
| Diagnostics (promotion rates, selectivity) | **PASS** |
| Controlled multi-seed experiment vs oracle & supervised | **PASS** |
| Utility learns real key/filler selectivity | **PASS** |
| Ready to proceed to Phase 4.2 (learned persistence) | **YES** |

## Files added

```
aac/models_diff.py          + AACDiffUtility class
run_phase41_utility.py      controlled experiment harness
test_phase41_utility.py     unit / mechanism tests (21/21 pass)
run_phase41_utility_results.json
PHASE41_UTILITY_RESULTS.md  this document
```

## Next phase (4.2)

Replace fixed `decay_p` / `decay_t` with learned / state-dependent
persistence:

```
λ_t = sigmoid(f_θ(x_t, S_t, T_t, P_{t-1}))
```

and run the fixed-λ vs learned-global-λ vs state-dependent-λ_t ablation
under the same protocol.
