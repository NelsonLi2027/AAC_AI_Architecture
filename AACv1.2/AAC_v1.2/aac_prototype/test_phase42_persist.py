"""Unit / mechanism tests for Phase 4.2 AACDiffPersist."""
import sys
import numpy as np
sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import AACDiffPersist, AdamTensor

PASS = 0
FAIL = 0

def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")

def test_modes_forward_and_grad():
    print("\n=== modes: forward + gradients ===")
    task = MQARTask(n_vocab=32, seed=0)
    ep = task.sample_episode(n_pairs=3, n_queries=2)
    for mode in ["fixed", "global", "state"]:
        m = AACDiffPersist(vocab_size=32, d_emb=16, d_h=16, persist_mode=mode, seed=1)
        m.opt = AdamTensor(m.params, lr=1e-2)
        loss, acc = m.forward_episode(ep, train=True)
        check(np.isfinite(loss), f"{mode}: loss finite ({loss:.3f})")
        if mode == "global":
            g = m.params["b_lambda"].grad
            check(g is not None and np.any(np.abs(g) > 0), f"{mode}: grad to b_lambda")
        if mode == "state":
            for name in ["Wl1", "Wl2", "bl2"]:
                g = m.params[name].grad
                check(g is not None and np.any(np.abs(g) > 0), f"{mode}: grad to {name}")
        # utility heads still train
        for name in ["Wv1", "Wv2", "W_keep"]:
            g = m.params[name].grad
            check(g is not None and np.any(np.abs(g) > 0), f"{mode}: grad to {name}")

def test_lambda_diagnostics():
    print("\n=== lambda diagnostics ===")
    task = MQARTask(n_vocab=32, seed=2)
    for mode in ["fixed", "global", "state"]:
        m = AACDiffPersist(vocab_size=32, d_emb=16, d_h=16, persist_mode=mode, seed=2)
        for _ in range(3):
            m.forward_episode(task.sample_episode(n_pairs=3, n_queries=2), train=False)
        s = m.summarize_diag()
        check(s["lambda_mean"] is not None, f"{mode}: lambda_mean logged")
        check(0.0 < s["lambda_mean"] <= 1.0, f"{mode}: lambda in (0,1] ({s['lambda_mean']:.4f})")
        check(len(m.last_lambda_values) > 0, f"{mode}: last_lambda_values non-empty")

def test_fixed_matches_utility_path():
    print("\n=== fixed mode uses constant decay_p ===")
    m = AACDiffPersist(vocab_size=16, d_emb=8, d_h=8, persist_mode="fixed",
                       decay_p=0.95, seed=0)
    check(m.persist_mode == "fixed", "mode is fixed")
    check(m.decay_p == 0.95, "decay_p stored")
    check("b_lambda" not in m.params, "no b_lambda in fixed")
    check("Wl1" not in m.params, "no Wl1 in fixed")

def test_no_oracle_training_target():
    print("\n=== no oracle training target ===")
    m = AACDiffPersist(vocab_size=16, d_emb=8, d_h=8, persist_mode="state", seed=0)
    check(not hasattr(m, "aux_gate_weight"), "no aux_gate_weight")
    check(not hasattr(m, "gate_mode"), "no gate_mode")

if __name__ == "__main__":
    test_modes_forward_and_grad()
    test_lambda_diagnostics()
    test_fixed_matches_utility_path()
    test_no_oracle_training_target()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
