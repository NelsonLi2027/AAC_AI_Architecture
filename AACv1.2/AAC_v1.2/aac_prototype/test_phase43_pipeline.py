"""Unit / mechanism tests for Phase 4.3 AACDiffPipeline."""
import sys
import numpy as np
sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import AACDiffPipeline, AdamTensor

PASS = FAIL = 0

def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")

def test_both_modes():
    print("\n=== direct & trace forward + grads ===")
    task = MQARTask(n_vocab=32, seed=0)
    ep = task.sample_episode(n_pairs=3, n_queries=2)
    for mode in ["direct", "trace"]:
        m = AACDiffPipeline(vocab_size=32, d_emb=16, d_h=16, pipeline_mode=mode,
                            persist_mode="global", seed=1)
        m.opt = AdamTensor(m.params, lr=1e-2)
        loss, acc = m.forward_episode(ep, train=True)
        check(np.isfinite(loss), f"{mode}: loss finite")
        for name in ["Wv1", "Wv2", "W_keep", "b_lambda"]:
            g = m.params[name].grad
            check(g is not None and np.any(np.abs(g) > 0), f"{mode}: grad to {name}")
        if mode == "trace":
            for name in ["W_tk", "W_tv"]:
                g = m.params[name].grad
                check(g is not None and np.any(np.abs(g) > 0), f"{mode}: grad to {name}")
            check("W_tk" in m.params and "W_tv" in m.params, f"{mode}: has T projections")
        else:
            check("W_tk" not in m.params, f"{mode}: no T projections")

def test_t_content_diagnostics():
    print("\n=== T content diagnostics (trace mode) ===")
    task = MQARTask(n_vocab=32, seed=2)
    m = AACDiffPipeline(vocab_size=32, d_emb=16, d_h=16, pipeline_mode="trace",
                        persist_mode="global", seed=2)
    for _ in range(3):
        m.forward_episode(task.sample_episode(n_pairs=3, n_queries=2), train=False)
    s = m.summarize_diag()
    check(s["tk_norm"] is not None and s["tk_norm"] > 0, f"tk_norm > 0 ({s['tk_norm']})")
    check(s["tv_norm"] is not None and s["tv_norm"] > 0, f"tv_norm > 0 ({s['tv_norm']})")
    check(s["trace_norm"] is not None, "trace_norm logged")
    check(s["mem_norm"] is not None, "mem_norm logged")

def test_t_neq_p_params():
    print("\n=== T and P are distinct mechanisms ===")
    m = AACDiffPipeline(vocab_size=16, d_emb=8, d_h=8, pipeline_mode="trace", seed=0)
    check(hasattr(m, "decay_t") and hasattr(m, "decay_p"), "separate decay_t / decay_p")
    check(m.pipeline_mode == "trace", "pipeline_mode=trace")
    # keep gate and promote gate are separate
    check("W_keep" in m.params and "Wv1" in m.params, "distinct keep vs value heads")

if __name__ == "__main__":
    test_both_modes()
    test_t_content_diagnostics()
    test_t_neq_p_params()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
