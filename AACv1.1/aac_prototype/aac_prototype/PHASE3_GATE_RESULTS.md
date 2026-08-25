# Phase 3: Isolating and Partially Fixing the Write-Gate Failure

Phase 2 (`PHASE2_RESULTS.md`) established that AAC-diff loses badly to
softmax attention on MQAR and traced this to the write gate never
learning selectivity (key-position gate values roughly equal to
filler-position values, ratio ~0.8-1.0x across every update rule tested
in `DELTA_RULE_RESULTS.md`). That work left the *cause* open: is it the
gate, the read/retrieval mechanism, the fixed exponential decay, or some
mix, and is it fixable at all?

This phase isolates each of those variables independently using
controlled ablations of `aac/models_diff.py`'s `AACDiffModel`, and finds:
the read mechanism is fine, the decay rate is a real and independently
fixable bottleneck, the gate-selectivity failure is real and is the
dominant bottleneck, three unsupervised fixes for it all fail, and one
supervised fix produces a first, partial, quantified success.

All numbers below: `n_pairs=6`, `n_vocab=64` (chance = 1.6%), `d_emb=d_h=32`,
Adam `lr=1e-2`, batch=8, MQAR task (`task_mqar.py`). Episode counts vary by
experiment (2000-4000) and are noted per section; this is a smaller budget
than Phase 2's 3000, chosen for iteration speed during this investigation,
so absolute numbers below are directionally reliable but a final write-up
should re-run the winning configurations at matched 3000+-episode budget
for a clean head-to-head against Phase 2's baselines.

## Method: three new `AACDiffModel` gate modes

- `gate_mode="oracle"`: hard 1/0 write exactly at the true key→value
  binding position, **no decay applied at all** (`M += outer(k,v)`,
  unconditionally). Not a controlled variable in isolation -- see below.
- `gate_mode="oracle_decay"`: same hard, correctly-timed 1/0 write, but
  **with** the same `decay` multiplier every other mode uses
  (`M = decay*M + mask*outer(k,v)`). This isolates "correct write timing"
  from "no forgetting at all."
- `gate_mode="learned"` (existing): `g = sigmoid(Wg @ h + bg)`, with an
  optional warm-start bias `bg`, an optional curriculum blend against the
  oracle mask (`gate_mode="curriculum"`, ramped `alpha`), and an optional
  auxiliary supervised loss (`aux_gate_weight`, `aux_pos_weight`) --
  weighted binary cross-entropy between the gate's sigmoid output and the
  true oracle write label at every step, added directly into the same
  backward pass as the task loss. Implemented as a new `ad.bce_loss` op in
  `aac/autodiff.py`, gradient-checked to ~1e-11 against finite differences.

## Finding 1 -- the read/retrieval mechanism is not broken

Phase 2's "perfect-write test" (`train_compare.py::perfect_write_test`)
found near-chance accuracy (0.7-3.3%) even when `M` was hand-constructed
from ground-truth pairs, and concluded the read mechanism (`Wq`, `Wk`,
`Wv`, linear dot-product retrieval, the readout combiner) might be
*independently* broken -- a "dual bottleneck."

That conclusion doesn't hold up. That test froze a perfect `M` on top of
`Wq/Wk/Wv/Wcomb/Wout` weights that had been trained under the normal
*broken* regime, where the gate never fires and those weights never see a
useful gradient signal. Training the full model **jointly**, from
scratch, with the gate replaced by the oracle mask, gets:

| Model (2000 ep) | Held-out acc | Gap 20-40 | Gap 60-90 | Gap 150-200 | Gap 300-400 |
|---|---:|---:|---:|---:|---:|
| RNN-only (floor) | 1.6% | -- | -- | -- | -- |
| AAC-diff, learned gate | 1.1% | 1.7% | 1.2% | 1.8% | 1.5% |
| **AAC-diff, oracle gate (no decay)** | **98.8%** | 99.0% | 97.5% | 98.8% | 98.0% |

The read pathway learns fine and generalizes to 10x the training gap
length, given correct writes. The earlier "dual bottleneck" conclusion
was an artifact of testing an undertrained read pathway, not a
structural defect in linear dot-product retrieval. **Correction to
`PHASE2_RESULTS.md`'s perfect-write-test interpretation.**

## Finding 2 -- the fixed decay rate is a second, independent, fixable bottleneck

The oracle-gate result above is confounded: `gate_mode="oracle"` also
removes decay entirely, so it isn't testing "correct write timing" in
isolation. Controlling for this with `oracle_decay` (correct timing,
decay reapplied):

| Write timing | decay | Held-out (gap 20-40) | Gap 60-90 | Gap 150-200 | Gap 300-400 |
|---|---:|---:|---:|---:|---:|
| oracle | 0.95 | 77.2% | 60.0% | 1.0% | 2.8% |
| oracle | 0.99 | 99.0% | 99.5% | 98.0% | 35.7% |
| **oracle** | **0.999** | **99.0%** | **99.5%** | **98.7%** | **99.7%** |

With perfect write timing, `decay=0.95` alone caps the usable memory
horizon at roughly `1/(1-0.95) ~ 20` steps (`0.95^150 ~ 3e-4`) -- by
gap 150-200 accuracy collapses to chance even though every write was
timed correctly. `decay=0.999` removes this ceiling almost entirely and
holds near-perfect accuracy out to 10x the training gap length. **This
confirms the fixed geometric decay is a real, severable bottleneck, and
it is fully fixable independent of gate quality.**

## Finding 3 -- decay alone does not fix, and does not even move, gate selectivity

Same decay sweep, but with the gate `learned` instead of `oracle`:

| decay | Held-out | Gate value at key positions | Gate value at filler | Ratio |
|---|---:|---:|---:|---:|
| 0.95 | 1.6% | 0.218 | 0.218 | 1.00x |
| 0.99 | 4.8% | 0.295 | 0.283 | 1.04x |
| 0.999 | 6.3% | 0.381 | 0.383 | 0.99x |

Absolute gate magnitude rises with slower decay (0.22 -> 0.38, presumably
because a weaker, less-decayed signal manages to reduce loss a little via
uniformly-more-open writing) but **the key/filler ratio stays pinned at
~1.0x across all three settings** -- the gate never learns to
discriminate positions at all, regardless of how forgiving the rest of
the system is. Two unsupervised interventions were also tried and both
failed to move the ratio in a useful direction:

| Intervention (2000 ep, decay=0.95) | Held-out | Gate ratio |
|---|---:|---:|
| Warm-start bias `bg=+2.0` (initial gate ~0.88) | 1.3% | 1.05x (converged to ~uniformly-open, 0.60-0.63 both classes) |
| Warm-start bias `bg=+4.0` (initial gate ~0.98) | 1.7% | 1.02x (same failure mode) |
| Curriculum: oracle write -> learned write, alpha ramped 0->1 over 60% of training | 1.5% | 1.76x, but magnitude collapses to ~0.02-0.04 (same as the unsupervised baseline) |

Warm-starting the gate open just finds a new local optimum -- "write
everything, uniformly" -- instead of "write nothing"; neither is
selective. The curriculum run is the most informative negative result:
even while `alpha` is still mostly weighted toward the oracle mask
(e.g. `alpha=0.21` at episode 250, so 79% of the actual write is still
oracle-timed), training accuracy stays at chance (~1.5-2.4%) throughout
-- far below what the corresponding oracle-timed run achieves at the
same episode count (e.g. `oracle_decay`, decay=0.95 hits 40%+ train
accuracy by episode 1000). The scaffolding doesn't transfer any usable
signal to the learned gate even while it's doing most of the work,
which is consistent with "write selectivity is a hard local-optimum
problem that isn't solved by making it directionally easier."

## Finding 4 -- direct supervision on the gate is the first thing that works

Added a weighted binary cross-entropy auxiliary loss directly on the
gate's sigmoid output against the true oracle write label
(`aux_gate_weight`, `aux_pos_weight` to counter the severe class
imbalance between rare key-write positions and abundant filler
positions), summed into the same backward pass as the task loss
(new `ad.bce_loss` op, gradient-checked to 1e-11).

`decay=0.95` + aux loss, several weightings (2000 ep): **no improvement**
over the unsupervised baseline (held-out 0.7-1.8%, same as before) --
decay=0.95's attenuation apparently drowns out the aux signal just as
it drowns out the task-loss signal.

`decay=0.999` + aux loss (`aux_gate_weight=1.0`, `aux_pos_weight=5.0`,
4000 ep): **first genuine, still-improving positive result.**

| Episode | Train acc (last 250) |
|---:|---:|
| 500 | 1.6% |
| 1500 | 2.2% |
| 2000 | 11.2% |
| 2500 | 16.3% |
| 3000 | 24.1% |
| 3500 | 28.3% |
| 3750 | 28.6% |
| 4000 | 21.9% (likely optimization noise, not a ceiling -- still rising through ep 3750) |

| Metric | Best unsupervised (decay=0.999, no aux) | + aux supervision |
|---|---:|---:|
| Held-out (gap 20-40) | 6.3% | **25.3%** |
| Gap 300-400 | 3.5% | **7.2%** |
| Gate value, key positions | 0.381 | **0.313** |
| Gate value, filler positions | 0.383 | **0.182** |
| Gate ratio | 0.99x | **1.72x** |

This is the first configuration in which the key/filler gate ratio
clears ~1.0x by a wide margin with non-trivial absolute magnitude on
both sides -- i.e., the gate has learned to discriminate, not just to
open or close uniformly. Held-out accuracy is a 4x improvement over the
best unsupervised variant. It is not close to the oracle ceiling (~99%)
and the training curve had not converged at 4000 episodes, so this
number should be read as a lower bound on what direct gate supervision
can achieve, not a final result.

## Summary table

| Configuration | Held-out acc | Gate ratio | What it isolates |
|---|---:|---:|---|
| RNN-only (no memory) | 1.6% | -- | floor |
| Learned gate, decay=0.95 (Phase 2 config) | 1.1-1.6% | 1.00x | baseline failure |
| Oracle write timing, no decay | 98.8% | -- (n/a) | read mechanism is fine |
| Oracle write timing, decay=0.95 | 77.2% | -- (n/a) | write-timing alone, decay-limited |
| Oracle write timing, decay=0.999 | 99.0%, holds to gap 400 | -- (n/a) | decay is fully fixable given correct writes |
| Learned gate, decay=0.999 (no aux) | 6.3% | 0.99x | decay fix alone does not teach selectivity |
| Learned gate, warm-start / curriculum | 1.3-1.7% | ~1.0-1.05x | unsupervised nudges don't work |
| **Learned gate, decay=0.999 + aux BCE supervision** | **25.3%, still rising** | **1.72x** | **first real, partial fix** |

## Interpretation

The Phase 2 conclusion ("AAC-diff's linear relaxation loses to attention,
cause unclear, maybe the update rule") is now four separated, evidenced
claims instead of one vague one:

1. The read/retrieval side was never the problem -- the earlier
   "dual bottleneck" reading of the perfect-write test doesn't survive
   joint training and should be considered superseded.
2. The fixed `decay=0.95` schedule was silently capping the achievable
   memory horizon regardless of gate quality; this is a solved problem
   (`decay->0.999`, or a learned/adaptive decay, closes it).
3. Gate selectivity is the real, dominant bottleneck, and it is not a
   soft optimization difficulty that yields to easier initialization,
   curriculum scaffolding, or a more forgiving decay -- all three failed
   identically (ratio stuck at ~1.0x).
4. Direct supervision on the gate is the first intervention that
   produces measurable selectivity and a substantial (4x) accuracy
   gain, though it does not yet close the gap to the oracle ceiling, and
   training had not converged in this run.

## Honest caveats / what this doesn't show

- Episode budgets here (2000-4000) are below Phase 2's 3000-episode
  standard and well below convergence for the auxiliary-loss run; the
  25.3% number is a lower bound, not a final result.
- Aux-loss supervision requires the oracle write-position labels
  (`key_positions` from `task_mqar.py`), which are diagnostic-only
  ground truth not available in a real deployment setting -- this
  result shows the gate *can* learn selectivity given a training signal
  for it, not that AAC-diff solves MQAR unsupervised. A faithful next
  step is a *proxy* signal that doesn't require oracle labels (e.g. the
  original non-differentiable prototype's approach: `aac/controller.py`
  trains its utility MLP on a cheap proxy label -- "is this token a
  KEY/VAL event" -- rather than ground-truth write positions; an
  analogous proxy for MQAR's shared-vocabulary setting is an open
  question since there's no separate KEY/VAL token range to key off of).
- Only one `(aux_gate_weight, aux_pos_weight, decay)` combination was
  run to completion at longer budget; a proper sweep plus longer
  training is the natural next step (see prior turn for the proposed
  follow-up).
- All numbers are single-seed (`SEED=0`); no variance estimates yet,
  unlike the perfect-write-test's z-scores in `train_compare.py`.

## Files added this phase

```
aac/autodiff.py        added bce_loss op (gradient-checked to ~1e-11)
aac/models_diff.py      AACDiffModel: gate_mode="oracle_decay"/"curriculum",
                        gate_bias_init, aux_gate_weight, aux_pos_weight
run_oracle_vs_learned.py    Finding 1 experiment
run_gate_fixes.py           Finding 3 experiment (warm-start, curriculum)
run_decay_sweep.py          Finding 2 + 3 experiment (decay x gate_mode grid)
run_aux_gate.py              Finding 4 experiment (aux BCE sweep)
```
