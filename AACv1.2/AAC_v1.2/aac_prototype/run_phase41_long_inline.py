import sys, time, json
import numpy as np
sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import AACDiffUtility, AdamTensor

D_EMB, D_H, N_VOCAB = 32, 32, 64
LR_START, LR_END = 1e-2, 1e-3
BATCH, CKPT, CEILING, DURATION = 8, 500, 12000, 10000
SEEDS = [0, 1, 2]
OUT = "run_phase41_utility_long_results.json"

def lr_schedule(ep, n_total):
    frac = min(1.0, max(0.0, ep / n_total))
    return LR_START + frac * (LR_END - LR_START)

def evaluate(model, n_episodes=150, seed=999):
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    accs = []
    for _ in range(n_episodes):
        ep = eval_task.sample_episode(n_pairs=6, n_queries=6, gap_len=(20, 40))
        _, acc = model.forward_episode(ep, train=False)
        accs.append(acc)
    return float(np.mean(accs))

def collect_diag(model, n_episodes=80, seed=777):
    if hasattr(model, "reset_diag"):
        model.reset_diag()
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    key_g, fill_g, all_g = [], [], []
    for _ in range(n_episodes):
        ep = eval_task.sample_episode(n_pairs=6, n_queries=6)
        model.forward_episode(ep, train=False)
        gates = getattr(model, "last_gate_values", [])
        oracle = set(kp + 1 for kp in ep["key_positions"] if kp + 1 < len(gates))
        for t, g in enumerate(gates):
            all_g.append(g)
            (key_g if t in oracle else fill_g).append(g)
    pk = float(np.mean(key_g)) if key_g else None
    pf = float(np.mean(fill_g)) if fill_g else None
    ratio = (pk / max(pf, 1e-8)) if (pk is not None and pf is not None) else None
    return {"promote_key": pk, "promote_filler": pf, "promote_ratio": ratio,
            "promote_rate": float(np.mean(all_g)) if all_g else None}

def check_converged(curve, window=4, tol_pp=1.5):
    if len(curve) < window:
        return False
    last = curve[-window:]
    for i in range(1, len(last)):
        if abs(last[i]["held_out"] - last[i-1]["held_out"]) * 100 > tol_pp:
            return False
    return True

all_results = []
for seed in SEEDS:
    sep = "=" * 72
    print(f"\n{sep}\nUtility seed={seed}\n{sep}", flush=True)
    task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    model = AACDiffUtility(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H,
                           decay_p=0.999, decay_t=0.9, seed=seed)
    model.opt = AdamTensor(model.params, lr=LR_START)
    train_accs, train_losses = [], []
    t0 = time.perf_counter()
    model.opt.zero_grad()
    curve = []
    converged = False
    ep = 0
    while ep < CEILING:
        ep += 1
        model.opt.lr = lr_schedule(ep, DURATION)
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = model.forward_episode(episode, train=True)
        train_accs.append(acc)
        train_losses.append(loss)
        if ep % BATCH == 0:
            model.opt.step()
            model.opt.zero_grad()
        if ep % CKPT == 0:
            ho = evaluate(model)
            d = collect_diag(model)
            point = {
                "ep": ep, "lr": float(model.opt.lr), "held_out": ho,
                "train_acc_window": float(np.mean(train_accs[-CKPT:])),
                "train_loss_window": float(np.mean(train_losses[-CKPT:])),
                **d,
            }
            curve.append(point)
            print(
                f"  [utility seed={seed}] ep {ep:6d}  lr={point['lr']:.5f}  "
                f"loss={point['train_loss_window']:.3f}  "
                f"train_acc={point['train_acc_window']:.3f}  "
                f"held_out={ho:.3f}  "
                f"prom_key={d['promote_key']:.3f}  "
                f"prom_fill={d['promote_filler']:.3f}  "
                f"ratio={d['promote_ratio']:.2f}x  "
                f"prom_rate={d['promote_rate']:.3f}  "
                f"({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
            partial = dict(config="utility", seed=seed, curve=curve,
                           converged=False, episodes_run=ep,
                           final_acc=ho, final_diag=point)
            payload = dict(results=all_results + [partial], summary={})
            with open(OUT, "w") as f:
                json.dump(payload, f, indent=2)
            if check_converged(curve):
                converged = True
                break
    r = dict(config="utility", seed=seed, curve=curve, converged=converged,
             final_acc=curve[-1]["held_out"] if curve else None,
             episodes_run=ep, final_diag=curve[-1] if curve else None)
    all_results.append(r)
    status = "CONVERGED" if converged else f"HIT CEILING {CEILING}"
    print(f"  -> seed={seed}  final_acc={r['final_acc']:.3f}  "
          f"episodes={ep}  {status}", flush=True)

finals = np.array([r["final_acc"] for r in all_results])
convs = [r["converged"] for r in all_results]
summary = dict(mean=float(finals.mean()), std=float(finals.std()),
               min=float(finals.min()), max=float(finals.max()),
               converged_count=int(sum(convs)), n_seeds=len(convs))
print(f"\nSUMMARY mean={summary['mean']:.3f} std={summary['std']:.3f} "
      f"min={summary['min']:.3f} max={summary['max']:.3f} "
      f"conv={summary['converged_count']}/{summary['n_seeds']}", flush=True)
with open(OUT, "w") as f:
    json.dump(dict(results=all_results, summary=summary), f, indent=2)
print(f"Saved {OUT}", flush=True)
