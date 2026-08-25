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
    recall loss, with no supervision telling it which tokens matter?"""
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    key_gates, filler_gates = [], []
    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=6, n_queries=6)
        model.forward_episode(episode, train=False)
        gates = model.last_gate_values
        key_pos = set(episode["key_positions"])
        for t, g in enumerate(gates):
            (key_gates if t in key_pos else filler_gates).append(g)
    return float(np.mean(key_gates)), float(np.mean(filler_gates))


if __name__ == "__main__":
    task = MQARTask(n_vocab=N_VOCAB, seed=SEED)
    n_episodes = 3000

    results = {}
    for cls in [RNNBaseline, AttentionBaseline, AACDiffModel]:
        print(f"\n=== Training {cls.name} ===")
        model, accs = train_model(cls, task, n_episodes)
        results[cls.__name__] = dict(model=model, accs=accs)

    print("\n=== Held-out evaluation (same distribution as training) ===")
    for name, r in results.items():
        acc = evaluate(r["model"], task)
        r["heldout_acc"] = acc
        print(f"  {r['model'].name:55s}  acc={acc:.3f}")

    print("\n=== Generalization: accuracy vs. noise-gap length "
          "(never seen this long during training) ===")
    gaps = [(20, 40), (60, 90), (150, 200), (300, 400)]
    print(f"  {'model':55s}  " + "  ".join(f"gap~{g[0]}-{g[1]:<4d}" for g in gaps))
    for name, r in results.items():
        accs_by_gap = [evaluate(r["model"], task, n_episodes=80, gap_len=g)
                       for g in gaps]
        r["gap_curve"] = accs_by_gap
        print(f"  {r['model'].name:55s}  " +
              "  ".join(f"{a:10.3f}" for a in accs_by_gap))

    print("\n=== AAC-diff write-gate diagnostic (emergent selectivity, "
          "no supervision) ===")
    aac_model = results["AACDiffModel"]["model"]
    key_g, filler_g = gate_diagnostic(aac_model, task)
    print(f"  mean write-gate value at KEY positions:    {key_g:.3f}")
    print(f"  mean write-gate value at FILLER positions: {filler_g:.3f}")
    print(f"  ratio: {key_g / max(filler_g, 1e-6):.2f}x "
          f"{'(learned to write more at informative tokens)' if key_g > filler_g * 1.2 else '(no clear emergent selectivity)'}")

    print("\n=== Summary ===")
    for name, r in results.items():
        print(f"  {r['model'].name:55s}  held-out={r['heldout_acc']:.3f}  "
              f"gap-300-400={r['gap_curve'][-1]:.3f}")
