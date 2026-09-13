"""Unit / mechanism tests for Phase 4.4 AACDiffLifecycle."""
import sys
import numpy as np
sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import AACDiffLifecycle, AdamTensor

PASS = FAIL = 0

def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")

def test_modes():
    print("\n=== modes forward + grads ===")
    task = MQARTask(n_vocab=32, seed=0)
    ep = task.sample_episode(n_pairs=3, n_queries=2)
    for mode in ["none", "reinforce", "reinforce_evict"]:
        m = AACDiffLifecycle(vocab_size=32, d_emb=16, d_h=16,
                             lifecycle_mode=mode, seed=1)
        m.opt = AdamTensor(m.params, lr=1e-2)
        loss, acc = m.forward_episode(ep, train=True)
        check(np.isfinite(loss), f"{mode}: loss finite")
        check(m.params["b_lambda"].grad is not None and
              np.any(np.abs(m.params["b_lambda"].grad) > 0), f"{mode}: grad λ")
        if mode != "none":
            check(m.params["W_reinf"].grad is not None and
                  np.any(np.abs(m.params["W_reinf"].grad) > 0), f"{mode}: grad reinf")
            check(m.summarize_diag()["reinforce_count"] > 0, f"{mode}: reinforce fired")
        if mode == "reinforce_evict":
            check(m.params["W_retain"].grad is not None and
                  np.any(np.abs(m.params["W_retain"].grad) > 0), f"{mode}: grad retain")
            check(m.summarize_diag()["retain_mean"] is not None, f"{mode}: retain logged")

def test_no_oracle():
    print("\n=== no oracle training target ===")
    m = AACDiffLifecycle(vocab_size=16, d_emb=8, d_h=8,
                         lifecycle_mode="reinforce_evict", seed=0)
    check(not hasattr(m, "aux_gate_weight"), "no aux_gate_weight")
    check(not hasattr(m, "gate_mode"), "no gate_mode")

if __name__ == "__main__":
    test_modes()
    test_no_oracle()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
