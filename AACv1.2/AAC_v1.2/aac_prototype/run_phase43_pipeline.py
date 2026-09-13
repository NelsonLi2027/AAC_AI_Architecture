"""
PHASE 4.3 — Differentiable T → P pipeline

Ablation under matched optimizer / schedule / seeds:

  A. direct — instantaneous outer(k,v) * p_promote → P
              (T only features the Value head; Phase 4.1/4.2 style)
  B. trace  — T stores soft key/value candidates; promotion writes
              transform(T) into P  (true T→P pipeline)

Both use learned global λ on P (Phase 4.2 winner) and learned utility
promotion (no oracle training target).

Usage:
  python3 run_phase43_pipeline.py
  python3 run_phase43_pipeline.py --seeds 0 1 --ceiling 8000
"""
import sys, time, argparse, json, os
import numpy as np

sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import AACDiffPipeline, AdamTensor

D_EMB, D_H, N_VOCAB = 32, 32, 64
LR_START, LR_END = 1e-2, 1e-3
BATCH = 8
DEFAULT_CEILING = 8000
DEFAULT_DURATION = 6000
DEFAULT_SEEDS = [0, 1]
ALL_MODES = ["direct", "trace"]
CKPT = 500
CONVERGE_WINDOW = 4
CONVERGE_TOL_PP = 1.5


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
    out = {
        "promote_key": pk, "promote_filler": pf,
        "promote_ratio": ratio,
        "promote_rate": float(np.mean(all_g)) if all_g else None,
    }
    if hasattr(model, "summarize_diag"):
        extra = model.summarize_diag()
        for k in ("lambda_mean", "trace_norm", "mem_norm", "tk_norm", "tv_norm",
                  "keep_key", "keep_filler"):
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


def _save(out_path, results):
    with open(out_path, "w") as f:
        json.dump(dict(results=results, summary={}), f, indent=2)


def run_one(mode, seed, duration, ceiling, ckpt, out_path, all_results):
    task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    model = AACDiffPipeline(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H,
                            pipeline_mode=mode, persist_mode="global",
                            decay_p=0.999, decay_t=0.9, seed=seed)
    model.opt = AdamTensor(model.params, lr=LR_START)
    train_accs, train_losses = [], []
    t0 = time.perf_counter()
    model.opt.zero_grad()
    curve = []
    converged = False
    ep = 0
    while ep < ceiling:
        ep += 1
        model.opt.lr = lr_schedule(ep, duration)
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = model.forward_episode(episode, train=True)
        train_accs.append(acc)
        train_losses.append(loss)
        if ep % BATCH == 0:
            model.opt.step()
            model.opt.zero_grad()
        if ep % ckpt == 0:
            ho = evaluate(model)
            d = collect_diag(model)
            point = {
                "ep": ep, "lr": float(model.opt.lr), "held_out": ho,
                "train_acc_window": float(np.mean(train_accs[-ckpt:])),
                "train_loss_window": float(np.mean(train_losses[-ckpt:])),
                **d,
            }
            curve.append(point)
            ratio = d.get("promote_ratio")
            ratio_s = f"{ratio:.2f}x" if ratio is not None else "n/a"
            lam = d.get("lambda_mean")
            lam_s = f"{lam:.4f}" if lam is not None else "n/a"
            print(
                f"  [{mode:6s} seed={seed}] ep {ep:6d}  lr={point['lr']:.5f}  "
                f"loss={point['train_loss_window']:.3f}  "
                f"train={point['train_acc_window']:.3f}  "
                f"held_out={ho:.3f}  λ={lam_s}  prom_ratio={ratio_s}  "
                f"({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
            partial = dict(config=mode, seed=seed, curve=curve, converged=False,
                           episodes_run=ep, final_acc=ho, final_diag=point)
            others = [r for r in all_results
                      if not (r.get("config") == mode and r.get("seed") == seed)]
            _save(out_path, others + [partial])
            # do not treat "stuck at chance" as convergence
            if check_converged(curve) and curve[-1]["held_out"] >= 0.20:
                converged = True
                break
    return dict(config=mode, seed=seed, curve=curve, converged=converged,
                final_acc=curve[-1]["held_out"] if curve else None,
                episodes_run=ep, final_diag=curve[-1] if curve else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--ceiling", type=int, default=DEFAULT_CEILING)
    parser.add_argument("--checkpoint-every", type=int, default=CKPT)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="run_phase43_pipeline_results.json")
    args = parser.parse_args()

    ckpt = args.checkpoint_every
    if args.quick:
        args.seeds = args.seeds[:1]
        args.duration = 400
        args.ceiling = 400
        ckpt = 100
        print("*** --quick ***\n")

    out_path = args.out
    all_results = []
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            all_results = [r for r in prev.get("results", [])
                           if r.get("converged") or r.get("episodes_run", 0) >= args.ceiling]
            print(f"Resuming; done: {[(r['config'], r['seed']) for r in all_results]}")
        except Exception:
            pass

    print(f"Modes: {args.modes}")
    print(f"Seeds: {args.seeds}")
    print(f"Schedule: linear {LR_START} → {LR_END} over {args.duration}")
    print(f"Ceiling: {args.ceiling}  ckpt every {ckpt}\n")

    for mode in args.modes:
        for seed in args.seeds:
            if any(r["config"] == mode and r["seed"] == seed for r in all_results):
                print(f"Skipping {mode} seed={seed}")
                continue
            print(f"\n{'='*72}\n{mode}  seed={seed}\n{'='*72}", flush=True)
            r = run_one(mode, seed, args.duration, args.ceiling, ckpt, out_path, all_results)
            all_results = [x for x in all_results
                           if not (x.get("config") == mode and x.get("seed") == seed)]
            all_results.append(r)
            status = "CONVERGED" if r["converged"] else f"CEILING {args.ceiling}"
            print(f"  -> {mode} seed={seed}  final_acc={r['final_acc']:.3f}  "
                  f"eps={r['episodes_run']}  {status}", flush=True)
            _save(out_path, all_results)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"{'mode':8s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}  conv")
    summary = {}
    for mode in args.modes:
        finals = np.array([r["final_acc"] for r in all_results if r["config"] == mode])
        if len(finals) == 0:
            continue
        convs = sum(1 for r in all_results if r["config"] == mode and r["converged"])
        print(f"{mode:8s} {finals.mean():8.3f} {finals.std():8.3f} "
              f"{finals.min():8.3f} {finals.max():8.3f}  {convs}/{len(finals)}")
        summary[mode] = dict(mean=float(finals.mean()), std=float(finals.std()),
                             min=float(finals.min()), max=float(finals.max()),
                             converged_count=convs, n_seeds=len(finals))
    with open(out_path, "w") as f:
        json.dump(dict(args=vars(args), results=all_results, summary=summary), f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
