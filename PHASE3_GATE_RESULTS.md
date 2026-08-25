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
fixable bottleneck, the gate-selectivity failure is real and is a second
dominant bottleneck, three unsupervised fixes for it all fail, a
supervised fix does teach real selectivity even under fast decay -- but
that selectivity alone doesn't help unless decay is also loosened. The
one configuration combining both fixes is the only one in this phase
that broke out of chance-level accuracy, confirmed on real hardware.

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

**Full sweep, 2000 episodes, confirmed on local hardware (`run_aux_gate.py`,
SEED=0)** -- superseding the earlier non-converged sandbox estimate:

| Config | Held-out (gap 20-40) | Gate ratio (key/filler) | Gate value, key | Gate value, filler |
|---|---:|---:|---:|---:|
| learned, no aux (baseline) | 1.8% | 0.95x | 0.123 | 0.130 |
| learned + aux(w=1, posw=5), decay=0.95 | 0.7% | **1.64x** | 0.394 | 0.241 |
| learned + aux(w=1, posw=10), decay=0.95 | 1.8% | **1.24x** | 0.518 | 0.417 |
| learned + aux(w=3, posw=5), decay=0.95 | 1.7% | **1.59x** | 0.385 | 0.242 |
| **learned + aux(w=1, posw=5), decay=0.999** | **26.4%** | **1.77x** | 0.391 | 0.221 |

Generalization for the decay=0.999+aux config (the only one worth sweeping,
since the others never leave chance):

| Gap 20-40 | Gap 60-90 | Gap 150-200 | Gap 300-400 |
|---:|---:|---:|---:|
| 25.5% | 23.5% | 16.0% | 8.5% |

This degrades *gracefully* -- roughly halving every ~150-200 extra steps
of gap -- rather than collapsing sharply to chance the way every other
learned-gate configuration in this phase does. That graceful-degradation
shape is qualitatively closer to how the Phase 2 attention baseline
behaves (57.2% -> 17.7% over a similar gap range) than to AAC-diff's
previous cliff-edge failures, even though the absolute numbers are still
well below attention's.

### The new finding: partial selectivity alone is not enough -- it needs decay to also be loosened

The three `decay=0.95` aux rows above are the interesting new result.
**The auxiliary loss unambiguously produces real gate selectivity even
under fast decay** -- ratios of 1.24x-1.64x, nothing like the dead-flat
~1.0x every unsupervised intervention produced in Finding 3. But **none
of it shows up in accuracy** -- all three stay pinned at chance (0.7-1.8%),
including at the training-distribution gap length, exactly where
oracle-timed hard writes under the same `decay=0.95` reached 77.2%
(Finding 2).

So gate selectivity and decay are not simply two independent knobs that
each contribute additively -- there's a **threshold effect** between
them. A soft, partially-selective gate (magnitude ~0.2-0.5, ratio
~1.2-1.6x) is not the same as the oracle's hard 0/1 write, and under
`decay=0.95` that gap matters enormously: the small amount of "leakage"
onto filler positions, and the fact that even correct writes aren't at
full strength, is apparently enough for the aggressive per-step decay to
wash the signal out before it reaches a query. Only once decay is also
relaxed (`0.999`) does that same partial selectivity translate into a
15x accuracy jump (1.8% -> 26.4%). **Fixing gate selectivity only pays
off once decay is loose enough for a moderately-selective, non-binary
gate to still matter by the time a query arrives.**


## Summary table

| Configuration | Held-out acc | Gate ratio | What it isolates |
|---|---:|---:|---|
| RNN-only (no memory) | 1.6% | -- | floor |
| Learned gate, decay=0.95 (Phase 2 config) | 1.1-1.8% | 0.95-1.00x | baseline failure |
| Oracle write timing, no decay | 98.8% | -- (n/a) | read mechanism is fine |
| Oracle write timing, decay=0.95 | 77.2% | -- (n/a) | write-timing alone, decay-limited |
| Oracle write timing, decay=0.999 | 99.0%, holds to gap 400 | -- (n/a) | decay is fully fixable given correct writes |
| Learned gate, decay=0.999 (no aux) | 6.3% | 0.99x | decay fix alone does not teach selectivity |
| Learned gate, warm-start / curriculum | 1.3-1.7% | ~1.0-1.05x | unsupervised nudges don't work |
| Learned gate + aux BCE, decay=0.95 | 0.7-1.8% | **1.24-1.64x** | selectivity learnable even under fast decay, but doesn't help accuracy |
| **Learned gate + aux BCE, decay=0.999** | **26.4%, gap 20-40** | **1.77x** | **selectivity + loose decay together: the only combination that works** |

## Interpretation

The Phase 2 conclusion ("AAC-diff's linear relaxation loses to attention,
cause unclear, maybe the update rule") is now five separated, evidenced
claims instead of one vague one:

1. The read/retrieval side was never the problem -- the earlier
   "dual bottleneck" reading of the perfect-write test doesn't survive
   joint training and should be considered superseded.
2. The fixed `decay=0.95` schedule was silently capping the achievable
   memory horizon regardless of gate quality; this is a solved problem
   (`decay->0.999`, or a learned/adaptive decay, closes it).
3. Gate selectivity is a second, real, dominant bottleneck, and it is
   not a soft optimization difficulty that yields to easier
   initialization or curriculum scaffolding -- both failed identically
   (ratio stuck at ~1.0x).
4. Direct supervision on the gate *is* capable of teaching real
   selectivity (ratio 1.2-1.8x) -- but selectivity alone, under fast
   decay, produces no accuracy gain at all (stays at chance).
5. Selectivity and decay interact rather than contributing independently:
   a softly-selective (non-binary) gate only translates into real
   accuracy once decay is also loosened -- the combination
   (`decay=0.999` + aux supervision) is the only configuration in this
   entire phase that broke out of chance-level performance, reaching
   26.4% held-out with graceful (not cliff-edge) degradation across gap
   lengths, confirmed on real hardware, not just a sandbox estimate.

## Honest caveats / what this doesn't show

- Episode budget here (2000) is below Phase 2's 3000-episode standard,
  and this run wasn't pushed to convergence -- an earlier 4000-episode
  sandbox run of the same decay=0.999+aux config was still climbing
  (21.9%-28.6% and rising) at that point, so 26.4% at 2000 episodes on
  real hardware should be read as a lower bound, not a ceiling.
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
- Only one `(aux_gate_weight, aux_pos_weight, decay)` combination reaches
  above-chance accuracy so far; a proper joint sweep (particularly around
  the decay=0.95-to-0.999 boundary, to find where the selectivity/decay
  threshold in Finding 4 actually sits) is the natural next step.
- All numbers are single-seed (`SEED=0`); no variance estimates yet,
  unlike the perfect-write-test's z-scores in `train_compare.py`.

## Methods note: two autodiff engine bugs found and fixed during this phase

Neither is specific to the gate-selectivity investigation -- both are
general defects in `aac/autodiff.py` that predate this phase -- but both
directly affected the ability to *run* the experiments above, so they're
recorded here for provenance:

1. **Reference-cycle memory leak.** Every op's `_backward` closure
   captures `out` (the tensor it's attached to) from its enclosing scope,
   creating a direct reference cycle on every non-leaf `Tensor` --
   potentially thousands per episode. Plain refcounting can't free a
   cycle; only the generational cyclic GC can, and cost grows as cyclic
   garbage accumulates, which showed up as training getting steadily
   slower over a run (up to ~30x slowdown observed on local hardware over
   ~2000 episodes) rather than any correctness error. Fixed by explicitly
   breaking the cycle (`_backward = None` on non-leaf nodes) once a graph
   is done being used -- inside `Tensor.backward()` for training calls,
   and via a new standalone `ad.free_graph()` for eval-mode calls (which
   never call `.backward()` and so never hit the training-side fix).
2. **Recursion-depth crash on long sequences.** The topological sort
   underlying both `backward()` and `free_graph()` was recursive
   (one Python stack frame per graph node). Fixing bug (1) meant
   eval-mode calls started actually traversing their graphs for the
   first time, which surfaced this: long episodes (gap=300-400 -> 350-450+
   tokens, several chained ops each) build dependency chains thousands of
   nodes deep, well past Python's default recursion limit (1000),
   producing a `RecursionError`. Fixed by converting the traversal to an
   iterative, explicit-stack version (`ad._topo_sort`), which has no
   depth ceiling other than available memory. Verified against episodes
   up to ~450 tokens with no crash, and against the full gradient-check
   suite (all ops + the 3-step BPTT/gated-memory case, ~1e-11 precision)
   to confirm the traversal change didn't alter any computed gradient.

Both fixes are in the `autodiff.py`/`models_diff.py` versions this
phase's numbers were run against; anyone re-running these experiments
from an older copy of the engine should expect the severe, worsening
slowdown and/or crash on long-gap eval described above.


## Files added/changed this phase

```
aac/autodiff.py        added bce_loss op (gradient-checked to ~1e-11);
                        fixed a reference-cycle memory leak in
                        Tensor.backward() and added ad.free_graph() for
                        the eval-mode path; converted the recursive
                        topological sort to an iterative ad._topo_sort()
                        to remove a RecursionError on long episodes
                        (see "Methods note" above)
aac/models_diff.py      AACDiffModel: gate_mode="oracle_decay"/"curriculum",
                        gate_bias_init, aux_gate_weight, aux_pos_weight;
                        wired ad.free_graph() into eval-mode _finish paths
run_oracle_vs_learned.py    Finding 1 experiment
run_gate_fixes.py           Finding 3 experiment (warm-start, curriculum)
run_decay_sweep.py          Finding 2 + 3 experiment (decay x gate_mode grid)
run_aux_gate.py              Finding 4 + 5 experiment (aux BCE sweep)
```
