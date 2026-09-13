"""
PHASE 4.1 extended training — utility policy only.

Goal: give AACDiffUtility enough episodes under the successful LR schedule
to decide whether the 46–67 % band is an optimization-time ceiling or an
architectural limit.

Logs every checkpoint (default every 500 ep):
  - held-out accuracy
  - promotion_key, promotion_filler, key/filler ratio
  - mean promotion rate (all positions)
  - train loss (last window)
  - train accuracy (last window)
  - episodes, lr

Multi-seed, convergence criterion, full curves saved to JSON.
"""
import sys, time, argparse, json
import numpy as np

sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import AACDiffUtility, AdamTensor

D_EMB, D_H = 32, 32
N_VOCAB = 64
LR_START = 1e-2
LR_END = 1e-3
BATCH = 8
CHECKPOINT_EVERY = 500
CONVERGE_WINDOW = 4          # stricter: 4 consecutive checkpoints
CONVERGE_TOL_PP = 1.5
DEFAULT_CEILING = 15000
DEFAULT_DURATION = 10000
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def lr_schedule(ep, n_total, lr_start=LR_START, lr_end=LR_END):
    frac = min(1.0, max(0.0, ep / n_total))
    return lr_start + frac * (lr_end - lr_start)


def evaluate(model, n_episodes=150, gap_len=(20, 40), seed=999):
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    accs = []
    for _ in range(n_episodes):
        ep = eval_task.sample_episode(n_pairs=6, n_queries=6, gap_len=gap_len)
        _, acc = model.forward_episode(ep, train=False)
        accs.append(acc)
    return float(np.mean(accs))


def collect_diagnostics(model, n_episodes=100, seed=777):
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
    prom_rate = float(np.mean(all_g)) if all_g else None
    out = {
        "promote_key": pk,
        "promote_filler": pf,
        "promote_ratio": ratio,
        "promote_rate": prom_rate,
    }
    if hasattr(model, "summarize_diag"):
        extra = model.summarize_diag()
        for k in ("keep_key", "keep_filler", "value_key", "value_filler",
                  "trace_norm", "mem_norm"):
            if extra.get(k) is not None:
                out[k] = extra[k]
    return out


def check_converged(curve, window=CONVERGE_WINDOW, tol_pp=CONVERGE_TOL_PP):
    if len(curve) < window:
        return False
    last = curve[-window:]
    for i in range(1, len(last)):
        if abs(last[i]["held_out"] - last[i - 1]["held_out"]) * 100 > tol_pp:
            return False
    return True


def run_one(seed, schedule_duration, ceiling, verbose=True):
    task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    model = AACDiffUtility(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H,
                           decay_p=0.999, decay_t=0.9, seed=seed)
    model.opt = AdamTensor(model.params, lr=LR_START)
    train_accs = []
    train_losses = []
    t0 = time.perf_counter()
    model.opt.zero_grad()
    curve = []
    converged = False
    ep = 0
    while ep < ceiling:
        ep += 1
        model.opt.lr = lr_schedule(ep, schedule_duration)
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = model.forward_episode(episode, train=True)
        train_accs.append(acc)
        train_losses.append(loss)
        if ep % BATCH == 0:
            model.opt.step()
            model.opt.zero_grad()
        if ep % CHECKPOINT_EVERY == 0:
            ho = evaluate(model, n_episodes=150, seed=999)
            d = collect_diagnostics(model)
            point = {
                "ep": ep,
                "lr": float(model.opt.lr),
                "held_out": ho,
                "train_acc_window": float(np.mean(train_accs[-CHECKPOINT_EVERY:])),
                "train_loss_window": float(np.mean(train_losses[-CHECKPOINT_EVERY:])),
                **d,
            }
            curve.append(point)
            if verbose:
                print(
                    f"  [utility seed={seed}] ep {ep:6d}  lr={point['lr']:.5f}  "
                    f"loss={point['train_loss_window']:.3f}  "
                    f"train_acc={point['train_acc_window']:.3f}  "
                    f"held_out={ho:.3f}  "
                    f"prom_key={d.get('promote_key'):.3f}  "
                    f"prom_fill={d.get('promote_filler'):.3f}  "
                    f"ratio={d.get('promote_ratio'):.2f}x  "
                    f"prom_rate={d.get('promote_rate'):.3f}  "
                    f"({time.perf_counter()-t0:.0f}s)"
                )
            if check_converged(curve):
                converged = True
                break
    return dict(
        config="utility",
        seed=seed,
        curve=curve,
        converged=converged,
        final_acc=curve[-1]["held_out"] if curve else None,
        episodes_run=ep,
        final_diag=curve[-1] if curve else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--ceiling", type=int, default=DEFAULT_CEILING)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="run_phase41_utility_long_results.json")
    args = parser.parse_args()

    ckpt = args.checkpoint_every
    if args.quick:
        args.seeds = args.seeds[:1]
        args.duration = 400
        args.ceiling = 400
        ckpt = 100
        print("*** --quick smoke test ***\n")

    # Make checkpoint interval available to run_one
    import sys
    this = sys.modules[__name__]
    this.CHECKPOINT_EVERY = ckpt

    print(f"Seeds:   {args.seeds}")
    print(f"Schedule: linear {LR_START} → {LR_END} over {args.duration}")
    print(f"Ceiling: {args.ceiling}  Checkpoint every {ckpt}")
    print(f"Convergence: {CONVERGE_WINDOW} consecutive ckpts within {CONVERGE_TOL_PP}pp\n")

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*72}\nUtility  seed={seed}\n{'='*72}")
        r = run_one(seed, args.duration, args.ceiling)
        all_results.append(r)
        status = "CONVERGED" if r["converged"] else f"HIT CEILING {args.ceiling}"
        print(f"  -> seed={seed}  final_acc={r['final_acc']:.3f}  "
              f"episodes={r['episodes_run']}  {status}")
        if r["final_diag"]:
            d = r["final_diag"]
            print(f"     prom_key={d.get('promote_key'):.3f}  "
                  f"prom_fill={d.get('promote_filler'):.3f}  "
                  f"ratio={d.get('promote_ratio'):.2f}x  "
                  f"prom_rate={d.get('promote_rate'):.3f}")

    finals = np.array([r["final_acc"] for r in all_results])
    convs = [r["converged"] for r in all_results]
    print(f"\n{'='*72}\nSUMMARY (utility, {len(args.seeds)} seeds)\n{'='*72}")
    print(f"mean={finals.mean():.3f}  std={finals.std():.3f}  "
          f"min={finals.min():.3f}  max={finals.max():.3f}  "
          f"converged={sum(convs)}/{len(convs)}")

    summary = dict(
        mean=float(finals.mean()), std=float(finals.std()),
        min=float(finals.min()), max=float(finals.max()),
        converged_count=int(sum(convs)), n_seeds=len(convs),
    )
    out = dict(args=vars(args), results=all_results, summary=summary)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull trajectories saved to {args.out}")


if __name__ == "__main__":
    main()
