# AAC Prototype

A working, runnable implementation of the AAC (Adaptive Associative
Computing) architecture described in `AAC.txt`, incorporating the
corrections from `AAC-corrections.md` (non-circular update order,
sign-corrected credit assignment, de-collided notation).

**Pure NumPy, zero dependencies.** No PyTorch was available in this
environment (no network access to install it), so this is a from-scratch
implementation, including a hand-derived MLP + Adam optimizer — there is
no autodiff framework underneath. `pip install numpy` is the only
requirement.

## Run it

```bash
python3 train.py              # train + evaluate on synthetic associative recall
python3 mechanism_checks.py   # isolated checks: routing speed, merge, eviction
```

## What it demonstrates

`train.py` trains on a synthetic **associative-recall** task: the model is
shown 5 `(KEY_i, VAL_j)` pairs separated by noise tokens, then a long noise
gap (much longer than the fast state's effective memory), then queried on
each `KEY_i` and must output the correct `VAL_j`. This is the standard
stress test for "does this architecture actually use long-range memory, or
is it just pattern-matching locally."

Actual output from a full run:

```
episode   100  train_acc(last 100)=0.404  writes=5  promotions=5  evictions=0  final|P|=5
episode  1500  train_acc(last 100)=0.944  writes=5  promotions=5  evictions=0  final|P|=5

accuracy with persistent memory (routing on):   0.939
accuracy with persistent memory (routing off):  0.939
accuracy with memory READS DISABLED (ablation): 0.048
  -> memory contributes +0.891 accuracy (89.1 pts)
```

Chance accuracy is 1/16 ≈ 0.0625 (16 possible values). Three things are
worth noting:

1. **Learning happens** — accuracy climbs from ~40% to ~94% as the
   controller's utility estimator and the readout head train.
2. **Memory stays sparse** — exactly 5 items get promoted for 5 pairs, no
   more, no duplicates. `|P_t| ≪ N` (section 17) holds without being
   hard-coded — it falls out of the write-gating and merge logic.
3. **Memory is load-bearing** — forcibly blinding memory reads (monkey-
   patching `PersistentMemory.read` to return zeros) collapses accuracy to
   near-chance. The architecture isn't solving the task some other way and
   carrying the memory system as dead weight.

`mechanism_checks.py` isolates three mechanisms that don't show up
clearly at the small scale above:

- **Hierarchical routing** (section 11): at `|P_t| = 4000`, k-means-routed
  retrieval is ~2.7x faster than flat search over all items.
- **Merge vs. competing hypotheses** (section 16): a near-duplicate key
  with the *same* value merges into one higher-confidence item; a
  near-duplicate key with a *different* value is kept as a **separate**
  item rather than silently overwritten.
- **Eviction** (section 15): deliberately weak items (`conf=0.1`) decay
  below threshold and get removed.

## File map

```
task.py               synthetic associative-recall episode generator
aac/mlp.py             hand-rolled MLP + Adam (the only gradient-trained parts)
aac/fast_state.py      S_t: diagonal + low-rank structured transition (section 5)
aac/trace.py           T_t: decaying evidence buffer, promotion (section 9)
aac/memory.py          P_t: write/read/routing/reinforce/evict/merge/credit-assignment
aac/controller.py      learned utility estimator + value-of-computation gating (7/8/19/20)
aac/model.py            wires it all together with the corrected, non-circular update order
train.py               training loop, held-out eval, ablation demo
mechanism_checks.py    isolated routing/merge/eviction checks
scaling_sweep.py       log-log routing-scaling fit + noise-gap generalization sweep
```

## A finding from the scaling sweep worth flagging

`scaling_sweep.py` fits a log-log slope to routing latency vs. memory size.
The **fixed** cluster count used elsewhere (`n_clusters=8`, the default)
gives slope ≈ 0.65 -- a real speedup over flat search, but still much
closer to linear (`O(M)`) than to the `O(log M)` the spec targets, because
each cluster ends up holding `M / 8` items regardless of `M`. Scaling the
cluster count with `sqrt(M)` instead (`memory.rebuild_routing(n_clusters=...)`
now takes an override) brings the slope down to ≈ 0.36, meaningfully
sub-linear but still short of true `O(log M)`, which would need genuine
multi-level (hierarchical-of-hierarchical) routing. This is called out
here rather than left implicit, since the original single-level, fixed-k
routing was not actually delivering the asymptotic scaling claimed.

## Phase 2: baselines, real end-to-end training, and a harder task

**See `PHASE2_RESULTS.md`.** That phase adds a hand-rolled, gradient-
checked autodiff engine (`aac/autodiff.py`), a harder shared-vocabulary
recall task with no pre-aligned embeddings (`task_mqar.py`), and a
matched-budget comparison against an RNN-only baseline and a softmax-
attention baseline (`aac/models_diff.py`, `train_compare.py`). Short
version: a plain differentiable relaxation of AAC's memory loses badly to
attention on the harder, fair benchmark, and the write-up explains why,
with reference to what would likely need to change (delta-rule updates)
to make it competitive.

## What's faithful to the spec vs. simplified

**Faithful:**
- The `Z_t = (S_t, T_t, P_t)` three-timescale split, with the corrected
  (non-circular) per-step ordering.
- Structured fast-state transition `A_t = D_t + L_t Q_t^⊤` (renamed from
  `P_t` to `L_t` per the notation fix).
- Memory item schema `(k_i, v_i, c_i, t_i, E_i)`, sparse write gating,
  decaying temporary trace with evidence-based promotion, associative
  softmax read, hierarchical (k-means) routing, confidence reinforcement
  and eviction, compatible-merge vs. competing-hypothesis preservation.
- Ablation-based causal credit assignment with the **corrected sign**:
  `C_i = L_ablated(i) − L_base` (positive when the memory genuinely helps).
- The value-of-computation gate (learned utility vs. fixed cost threshold)
  and adaptive multi-step reasoning driven by output-entropy.

**Simplified, and why:**
- **No end-to-end backprop through time.** `S_t`'s weights are a fixed
  random ("echo state network") reservoir rather than learned via BPTT.
  Only the controller's utility MLP and the output readout MLP are
  trained, with plain per-step SGD (Adam). This was a deliberate choice
  given no autodiff framework was available — implementing a correct
  hand-rolled BPTT through the full recurrent + memory system was out of
  scope for a prototype. The fast state still does real, useful
  compression (it's what the readout combines with retrieved memory); it
  just isn't gradient-trained here.
- **Token identity embeddings are fixed, not learned.** `KEY_i` and
  `QUERY_i` share a pre-assigned identity vector so the memory can match
  a later query to an earlier key. In a full system this binding would
  itself be learned from raw tokens; here it's assumed, so the prototype's
  learning is concentrated on the architecture's actual novel
  contribution — *when* to write, promote, retrieve, and how much extra
  computation to spend — rather than on relearning word embeddings.
- **Utility estimator is trained with a proxy label** (`1` for KEY/VAL
  events, `0` for filler/query), not the true counterfactual `U(e_t) =
  E[L_without − L_with]` from section 7, which isn't cheaply observable
  online. The ablation-based estimator in `memory.py` *does* compute the
  real counterfactual, but only as a periodic, more expensive credit-
  assignment pass (as section 21 itself implies it should be) — that
  part is not simplified.
- **One memory per episode**, reset at the start of each training episode,
  rather than one memory persisting across an unbounded stream. The
  eviction/cap logic (`max_items`) is there and tested (see
  `mechanism_checks.py`) for the case where it would matter.

If you want the next iteration to have genuine end-to-end learning of the
fast-state transition and token embeddings, the natural path is porting
`aac/fast_state.py` and the embedding tables to PyTorch (or JAX) once one
is available, and implementing `F_S`, `E`, and `Q` as small trainable
networks with BPTT — the rest of the architecture (memory, controller,
trace) carries over largely unchanged since it's already organized as
discrete, inspectable operations rather than a monolithic differentiable
graph.
