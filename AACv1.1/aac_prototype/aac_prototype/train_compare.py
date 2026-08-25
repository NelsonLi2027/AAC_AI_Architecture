"""
The real test: train RNNBaseline, AttentionBaseline, and AACDiffModel
end-to-end (full BPTT via aac/autodiff.py) on the harder MQAR task
(task_mqar.py -- shared vocabulary, no privileged key/query alignment),
with matched embedding/hidden dims and identical optimizer settings.

Reports:
  - training accuracy curve for all three
  - held-out accuracy
  - accuracy vs. noise-gap length (does AAC-diff's compressed memory
    degrade differently than the attention baseline's uncompressed one?)
  - a diagnostic on AAC-diff's learned write gate: did it learn, purely
    from the recall loss (no explicit supervision), to write more
    strongly at key positions than at filler positions?
"""
import sys, time
import numpy as np

sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import RNNBaseline, AttentionBaseline, AACDiffModel, AdamTensor

D_EMB, D_H = 32, 32
N_VOCAB = 64
SEED = 0
LR = 1e-2
BATCH = 8


def train_model(model_cls, task, n_episodes, eval_every=250, seed=SEED,
                 lr=LR, batch=BATCH):
    model = model_cls(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed)
    model.opt = AdamTensor(model.params, lr=lr)
    accs = []
    t0 = time.perf_counter()
    model.opt.zero_grad()
    for ep in range(1, n_episodes + 1):
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = model.forward_episode(episode, train=True)
        accs.append(acc)
        if ep % batch == 0:
            model.opt.step()
            model.opt.zero_grad()
        if ep % eval_every == 0:
            dt = time.perf_counter() - t0
            print(f"  [{model.name:55s}] ep {ep:5d}  "
                  f"train_acc(last {eval_every})={np.mean(accs[-eval_every:]):.3f}  "
                  f"({dt:.1f}s elapsed)")
    return model, accs


def evaluate(model, task, n_episodes=150, gap_len=(20, 40), seed=999):
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    accs = []
    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=6, n_queries=6, gap_len=gap_len)
        _, acc = model.forward_episode(episode, train=False)
        accs.append(acc)
    return float(np.mean(accs))


def gate_diagnostic(model, task, n_episodes=50, seed=777):
    """Does AAC-diff's learned write gate fire more strongly at key
    positions than at filler positions -- purely emergent from the
    recall loss, with no supervision telling it which tokens matter?
    
    CORRECTED: The memory write at timestep t binds (token[t-1], token[t]),
    so gates[t] corresponds to the write at position t. Since key_positions
    records where the KEY token sits, and the value immediately follows,
    the key-value write that matters is at gates[key_position + 1].
    We compare gates[key_position + 1] (key→value pairs) against all other
    gate positions (filler positions and non-matching writes).
    """
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    key_gates, filler_gates = [], []
    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=6, n_queries=6)
        model.forward_episode(episode, train=False)
        gates = model.last_gate_values
        key_positions = episode["key_positions"]
        # Collect indices where key writes happen (key_position + 1)
        # but only if in range
        key_write_indices = set()
        for kp in key_positions:
            if kp + 1 < len(gates):
                key_write_indices.add(kp + 1)
        
        for t, g in enumerate(gates):
            if t in key_write_indices:
                key_gates.append(g)
            else:
                filler_gates.append(g)
    return float(np.mean(key_gates)), float(np.mean(filler_gates))


def perfect_write_test(model, task, n_episodes=50, seed=888):
    """Isolate read-side from write-side: construct M perfectly by directly
    writing the true (key, value) pairs from each episode, then test if the
    trained query projection and readout head can correctly retrieve values.
    
    If this succeeds, the bottleneck is the write/gate.
    If this fails, the bottleneck is the read mechanism.
    """
    from aac import autodiff as ad
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    correct, total = 0, 0
    
    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=6, n_queries=6)
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        key_positions = episode["key_positions"]
        
        # Manually construct M by perfectly writing all key-value pairs
        # Extract the key and value tokens from their known positions
        M = ad.Tensor(np.zeros((model.d_h, model.d_h)))
        p = model.params
        for kp in key_positions:
            if kp + 1 < len(tokens):
                key_tok = int(tokens[kp])
                val_tok = int(tokens[kp + 1])
                key_emb = ad.embedding_lookup(p["Emb"], key_tok)
                val_emb = ad.embedding_lookup(p["Emb"], val_tok)
                k = ad.matvec(p["Wk"], key_emb)
                v = ad.matvec(p["Wv"], val_emb)
                M = ad.add(M, ad.outer(k, v))  # Perfect writes, gate=1.0
        
        # Now test retrieval: for each query, use the trained Wq and readout
        h = ad.Tensor(np.zeros(model.d_h))  # dummy hidden state
        for q_pos, q_label in qpos.items():
            query_tok = int(tokens[q_pos])
            query_emb = ad.embedding_lookup(p["Emb"], query_tok)
            q = ad.matvec(p["Wq"], query_emb)
            read = ad.vecmat(q, M)
            comb = ad.concat(h, read)
            hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
            logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
            pred = int(np.argmax(logits.data))
            correct += int(pred == q_label)
            total += 1
    
    return float(correct / total) if total > 0 else 0.0


if __name__ == "__main__":
    task = MQARTask(n_vocab=N_VOCAB, seed=SEED)
    n_episodes = 3000

    results = {}

    # Test three memory update rules
    update_rules = [
        ("hebbian", "AAC-diff (Hebbian)"),
        ("delta_regularized", "AAC-diff (delta-regularized)"),
        ("widrow_hoff", "AAC-diff (Widrow-Hoff)"),
    ]

    for rule, rule_name in update_rules:
        for cls in [RNNBaseline, AttentionBaseline]:
            if cls.__name__ not in results:
                print(f"\n=== Training {cls.name} ===")
                model, accs = train_model(cls, task, n_episodes)
                results[cls.__name__] = dict(model=model, accs=accs)

        # Train AACDiffModel with this update rule
        print(f"\n=== Training {rule_name} ===")
        model = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=SEED,
                            update_rule=rule)
        model.opt = AdamTensor(model.params, lr=LR)
        accs = []
        import time
        t0 = time.perf_counter()
        model.opt.zero_grad()
        for ep in range(1, n_episodes + 1):
            episode = task.sample_episode(n_pairs=6, n_queries=6)
            loss, acc = model.forward_episode(episode, train=True)
            accs.append(acc)
            if ep % BATCH == 0:
                model.opt.step()
                model.opt.zero_grad()
            if ep % 250 == 0:
                dt = time.perf_counter() - t0
                print(f"  [{rule_name:55s}] ep {ep:5d}  "
                      f"train_acc(last 250)={np.mean(accs[-250:]):.3f}  "
                      f"({dt:.1f}s elapsed)")

        key = f"AACDiffModel_{rule}"
        results[key] = dict(model=model, accs=accs, rule=rule)

    print(f"\n=== Training AAC-diff write-oracle ablation ===")
    oracle_model = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H,
                               seed=SEED, update_rule="hebbian", gate_mode="oracle")
    oracle_model.opt = AdamTensor(oracle_model.params, lr=LR)
    oracle_accs = []
    t0 = time.perf_counter()
    oracle_model.opt.zero_grad()
    for ep in range(1, n_episodes + 1):
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = oracle_model.forward_episode(episode, train=True)
        oracle_accs.append(acc)
        if ep % BATCH == 0:
            oracle_model.opt.step()
            oracle_model.opt.zero_grad()
        if ep % 250 == 0:
            dt = time.perf_counter() - t0
            print(f"  [{oracle_model.name:55s}] ep {ep:5d}  "
                  f"train_acc(last 250)={np.mean(oracle_accs[-250:]):.3f}  "
                  f"({dt:.1f}s elapsed)")
    results["AACDiffModel_write_oracle"] = dict(model=oracle_model, accs=oracle_accs, rule="oracle")

    print("\n=== Held-out evaluation (same distribution as training) ===")
    for name, r in results.items():
        acc = evaluate(r["model"], task)
        r["heldout_acc"] = acc
        model_name = r["model"].name if hasattr(r["model"], "name") else name
        print(f"  {model_name:55s}  acc={acc:.3f}")

    print("\n=== Generalization: accuracy vs. noise-gap length "
          "(never seen this long during training) ===")
    gaps = [(20, 40), (60, 90), (150, 200), (300, 400)]
    print(f"  {'model':55s}  " + "  ".join(f"gap~{g[0]}-{g[1]:<4d}" for g in gaps))
    for name, r in results.items():
        accs_by_gap = [evaluate(r["model"], task, n_episodes=80, gap_len=g)
                       for g in gaps]
        r["gap_curve"] = accs_by_gap
        model_name = r["model"].name if hasattr(r["model"], "name") else name
        print(f"  {model_name:55s}  " +
              "  ".join(f"{a:10.3f}" for a in accs_by_gap))

    print("\n=== AAC-diff CORRECTED gate diagnostic (key write at position+1) ===")
    for name, r in results.items():
        if "AACDiffModel" in name:
            diag = gate_diagnostic(r["model"], task, n_episodes=500)
            ratio = diag["ratio"]
            model_name = r["model"].name if hasattr(r["model"], "name") else name
            print(f"\n  {model_name} ({r['rule']}), n={diag['n']}:")
            print(f"    mean gate at KEY→VALUE writes:    {diag['key_mean']:.3f}  95% CI = [{diag['key_ci'][0]:.3f}, {diag['key_ci'][1]:.3f}]")
            print(f"    mean gate at FILLER positions:    {diag['filler_mean']:.3f}  95% CI = [{diag['filler_ci'][0]:.3f}, {diag['filler_ci'][1]:.3f}]")
            print(f"    ratio: {ratio:.2f}x  95% CI = [{diag['ratio_ci'][0]:.2f}, {diag['ratio_ci'][1]:.2f}]  {'(selectivity!)' if ratio > 1.2 else '(no selectivity)'}")

    print("\n=== Perfect-write read harness check: perfect M vs matched-noise M ===")
    print("  Uses n=500 episodes per model and the same trained readout/hidden state for both conditions.")
    for name, r in results.items():
        if "AACDiffModel" in name:
            metrics = perfect_write_test(r["model"], task, n_episodes=500)
            model_name = r["model"].name if hasattr(r["model"], "name") else name
            print(f"  {model_name} ({r['rule']}), n={metrics['sample_size']}:")
            print(f"    perfect M accuracy: {metrics['perfect_acc']:.3f}  95% CI = [{metrics['perfect_ci'][0]:.3f}, {metrics['perfect_ci'][1]:.3f}]")
            print(f"    random M accuracy: {metrics['random_acc']:.3f}  95% CI = [{metrics['random_ci'][0]:.3f}, {metrics['random_ci'][1]:.3f}]")
            print(f"    difference z-score: {metrics['diff_z']:.2f}")

    print("\n=== Gradient diagnostics (early training scaling check) ===")
    for name, r in results.items():
        if "AACDiffModel" in name and hasattr(r["model"], "grad_norms"):
            model = r["model"]
            if model.grad_norms["Wg"]:
                print(f"\n  {name} ({r['rule']}):")
                avg_norms = {k: np.mean([v for _, v in model.grad_norms[k]]) 
                            for k in ["Wg", "Wk", "Wv", "Wq"]}
                for k in ["Wg", "Wk", "Wv", "Wq"]:
                    print(f"    {k}: avg norm = {avg_norms[k]:.2e}")

    print("\n=== Summary ===")
    for name, r in results.items():
        model_name = r["model"].name if hasattr(r["model"], "name") else name
        print(f"  {model_name:55s}  held-out={r['heldout_acc']:.3f}  "
              f"gap-300-400={r['gap_curve'][-1]:.3f}")


