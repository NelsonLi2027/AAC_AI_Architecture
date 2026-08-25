# Delta-Rule Implementation Results

## Summary
Implemented two delta-rule variants for AAC-diff's memory mechanism in addition to the original Hebbian update. Neither variant improved performance on the MQAR benchmark, confirming that **the bottleneck is not the memory update rule, but the model's failure to learn selective writing**.

## Implementations Tested

### 1. Regularized Delta-Rule (Original Approach)
**Memory update:**
$$M_t = \text{decay} \cdot M_{t-1} + g_t \cdot \text{outer}(k, v) - g_t \cdot \beta \cdot \text{outer}(k, k^T M_{t-1})$$

**Intuition:** Subtract the projection of the old memory along $k$ before adding the new association.

**Result:** No improvement. Still ~2.2% accuracy.

### 2. Widrow-Hoff Delta-Rule (Second Approach)  
**Memory update:**
$$M_t = M_{t-1} + g_t \cdot \text{outer}(k, v - k^T M_{t-1})$$

**Intuition:** Direct error-correcting rule - update proportional to mismatch between target value $v$ and retrieved value $k^T M_{t-1}$.

**Result:** No improvement. Still ~0.8% accuracy (slightly worse).

## Benchmark Results (MQAR Task)

| Model | Held-out Accuracy | Gap 20-40 | Gap 300-400 |
|-------|-------------------|-----------|------------|
| RNN-only | 1.0% | 1.7% | 1.0% |
| **Attention** | **57.2%** | **57.1%** | **17.7%** |
| Hebbian (original) | 2.2% | 2.3% | 1.5% |
| Regularized delta-rule | 2.2% | 2.3% | 1.5% |
| Widrow-Hoff delta-rule | 0.8% | 0.6% | 2.9% |

## Key Finding: Gate Selectivity Failure

The **write gate never learns to be selective**:

| Update Rule | Key Positions | Filler Positions | Ratio |
|-------------|--------------|-----------------|-------|
| Hebbian | 0.371 | 0.437 | 0.85x |
| Regularized delta-rule | 0.371 | 0.437 | 0.85x |
| Widrow-Hoff delta-rule | 0.412 | 0.493 | 0.84x |

**Target:** Gate should be higher at KEY positions (ratio > 1.0), but all variants have ratio < 0.9x.

The model writes MORE at filler positions, which explains why it can't learn the task.

## Why Delta-Rule Didn't Fix the Problem

Delta-rule updates require that the model makes some correct associations, and the rule then refines them. However:

1. **The model never learns what to write in the first place** - the gate doesn't distinguish keys from fillers
2. **Attention learns via softmax's built-in selectivity** - its retrieval head automatically attends more to relevant keys
3. **AAC-diff's gate has no such structure** - it's just `sigmoid(W_g @ h)`, which doesn't get any "supervision" to learn selectivity

## Conclusion

Phase 2 correctly identified that the memory update rule wasn't the primary bottleneck. The continuous relaxation of AAC's memory loses the discrete routing logic that made selective writing possible. The delta-rule improvements help in systems where selective writing already occurs (like DeltaNet), but here the model never learns selectivity in the first place.

**Next step (if pursuing this further):**
- Revisit AAC's discrete write/promote/evict decisions
- Or: add an explicit supervision signal to the write gate to learn selectivity
- Or: use a hybrid architecture (local softmax attention + recurrent state)

The experiment successfully validates Phase 2's conclusion: **a literal continuous relaxation of AAC's memory is not competitive with softmax attention on this task**.

## Files Modified

1. **aac/autodiff.py** - Added `sub` operation with gradient checking
2. **aac/models_diff.py** - Added delta-rule memory update implementations
3. **test_autodiff.py** - Added tests for subtraction and delta-rule BPTT (all pass ~1e-11 precision)

All autodiff changes are verified and numerically correct.

---

## Corrected Diagnostic (Off-by-One Bug Fix)

### The Bug Discovery
During Phase 4 re-analysis, an **off-by-one indexing bug** was identified in the original gate selectivity diagnostic:
- **Original error:** Compared gate values at KEY token positions (where the key embedding sits) against filler positions
- **Actual timing:** The memory write happens ONE position LATER (when the value token appears), so the gate should be evaluated at position `key_position + 1`, not `key_position`
- **Impact:** The original diagnostic measured the wrong gates, potentially masking or distorting the true selectivity pattern

### Corrected Gate Diagnostic Results (with fixed indexing)

| Update Rule | Gates at KEY→VALUE writes | Gates at FILLER positions | Ratio |
|-------------|--------------------------|--------------------------|-------|
| Hebbian | 0.208 | 0.227 | **0.92x** |
| Regularized delta-rule | 0.469 | 0.463 | **1.01x** |
| Widrow-Hoff delta-rule | 0.464 | 0.483 | **0.96x** |

**Interpretation:**
- Hebbian: Still shows **no selectivity** (0.92x - gates LOWER at key positions)
- Regularized delta-rule: Nearly **neutral** (1.01x - effectively random writes)
- Widrow-Hoff: Slight **anti-selectivity** (0.96x - gates LOWER at key positions)

Even with corrected indexing, none of the three update rules achieve selective writing. This confirms the original conclusion: **the bottleneck is gate learning, not the update rule**.

### Perfect-Write Test Results (Read Mechanism Isolation)

Constructed perfect memory matrices by directly writing ground-truth (key, value) pairs with gate = 1.0, then tested if the trained query head could retrieve values:

| Update Rule | Perfect-Write Accuracy | Chance Baseline | Status |
|-------------|------------------------|-----------------|--------|
| Hebbian | 0.7% | 1.6% | **Below chance** |
| Regularized delta-rule | 3.3% | 1.6% | Barely above chance |
| Widrow-Hoff delta-rule | 1.3% | 1.6% | **Below chance** |

**Key Finding:** Even with perfect memory construction, the models **cannot reliably retrieve values**. This reveals a **dual bottleneck**:
1. ✗ Write gate doesn't learn selectivity  
2. ✗ Read head (Wq + retrieval) is also broken

The read mechanism failure is a **critical discovery** - it suggests the problem isn't just gate learning, but the fundamental architecture's ability to implement retrieval from the fixed-size associative memory.

### Gradient Diagnostics (Early Training Scaling)

Logged gradient norms for first 500 training episodes to detect scaling anomalies:

| Parameter | Hebbian | Regularized | Widrow-Hoff |
|-----------|---------|-------------|-------------|
| **Wg** (gate) | 2.63e-1 | 1.41e+0 | 1.45e+0 |
| **Wk** (key) | 9.27e-1 | 2.35e+0 | **5.80e+0** |
| **Wv** (value) | 8.55e-1 | 2.60e+0 | 2.60e+0 |
| **Wq** (query) | 8.81e-1 | 2.37e+0 | 2.87e+0 |

**Observation:** Widrow-Hoff's **Wk has 6.3x larger gradients** (5.80e+0 vs 9.27e-1) compared to Hebbian. This could indicate:
- Gradient explosion or instability in the key projection
- Potential numerical issues in the Widrow-Hoff update rule
- Misalignment between the error term and the update direction

### Revised Conclusion

The corrected diagnostics **reinforce and deepen** the original Phase 2 findings:

1. **Gate selectivity remains unlearned** across all three update rules
2. **Read mechanism fundamentally broken** - even perfect writes can't be retrieved
3. **Widrow-Hoff scaling issues** - much larger gradients suggest potential optimization instability
4. **Delta-rule variants unhelpful** because they assume learned associations that never materialize

The failure is **architectural**, not algorithmic. The continuous relaxation of AAC's memory requires the model to learn:
- Selective writes (gate learns to activate only on keys)
- Selective reads (query learns to extract the right entries)

With standard gradient descent and no additional structure, neither of these emerge naturally. Attention succeeds because softmax provides built-in selectivity; AAC-diff's bare sigmoid gate has no such guarantee.

### Implications

This corrected analysis suggests **three avenues for fixing AAC-diff** (in priority order):
1. **Add read structure:** Implement local softmax attention over the memory slots (similar to Transformers) rather than full dense retrieval
2. **Add write structure:** Impose sparsity/discreteness on the write gate via concrete dropout, masking, or gating
3. **Add supervision:** Supervise the gate to activate selectively, or use curriculum learning to bootstrap the behavior
