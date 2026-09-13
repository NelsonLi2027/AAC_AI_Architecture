"""
PHASE 4.1 — Learned write / promotion mechanism

Controlled comparison of three write mechanisms under identical
optimizer, schedule, task, and seeds:

  A. oracle gate          (hard ground-truth write positions)
  B. learned + supervised (aux BCE on oracle labels — Phase 3 style)
  C. learned utility      (AACDiffUtility — no oracle as training target;
                           promotion driven only by task loss via Value_θ)

All runs use the linear LR schedule that previously rescued training
(1e-2 → 1e-3 over 9000 episodes) and the same convergence criterion
as run_lr_schedule.py.

Diagnostics logged every checkpoint:
  - promotion rate on key vs filler positions
  - keep-in-trace rate on key vs filler
  - value-score means
  - temporary-trace / persistent-memory norms
  - promotion selectivity ratio (key / filler)

Usage:
    python3 run_phase41_utility.py
    python3 run_phase41_utility.py --seeds 0 1 2 --ceiling 6000
    python3 run_phase41_utility.py --quick
"""
import sys, time, argparse, json
import numpy as np

sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import (
    AACDiffModel, AACDiffUtility, AdamTensor, RNNBaseline, AttentionBaseline
)

D_EMB, D_H = 32, 32
N_VOCAB = 64
LR_START = 1e-2
LR_END = 1e-3
BATCH = 8
CHECKPOINT_EVERY = 1000
CONVERGE_WINDOW = 3
CONVERGE_TOL_PP = 2.0
DEFAULT_CEILING = 12000
DEFAULT_SEEDS = [0, 1, 2]
ALL_CONFIGS = ["oracle", "supervised", "utility"]


def lr_schedule(ep, n_total, lr_start=LR_START, lr_end=LR_END):
    frac = min(1.0, max(0.0, ep / n_total))
    return lr_start + frac * (lr_end - lr_start)


def make_model(config_name, seed):
    if config_name == "oracle":
        m = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed,
                          gate_mode="oracle_decay", decay=0.999)
    elif config_name == "supervised":
        m = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed,
                          gate_mode="learned", decay=0.999,
                          aux_gate_weight=1.0, aux_pos_weight=5.0)
    elif config_name == "utility":
        m = AACDiffUtility(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H,
                            decay_p=0.999, decay_t=0.9, seed=seed)
    else:
        raise ValueError(config_name)
    m.opt = AdamTensor(m.params, lr=LR_START)
    return m


def evaluate(model, n_episodes=150, gap_len=(20, 40), seed=999):
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    accs = []
    for _ in range(n_episodes):
        ep = eval_task.sample_episode(n_pairs=6, n_queries=6, gap_len=gap_len)
        _, acc = model.forward_episode(ep, train=False)
        accs.append(acc)
    return float(np.mean(accs))


def collect_diagnostics(model, n_episodes=80, seed=777):
    """Run a few held-out episodes purely for diagnostic logging.
    Works for both AACDiffModel (gate_values) and AACDiffUtility (full diag)."""
    if hasattr(model, "reset_diag"):
        model.reset_diag()
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    key_g, fill_g = [], []
    for _ in range(n_episodes):
        ep = eval_task.sample_episode(n_pairs=6, n_queries=6)
        model.forward_episode(ep, train=False)
        gates = getattr(model, "last_gate_values", [])
        oracle = set(kp + 1 for kp in ep["key_positions"] if kp + 1 < len(gates))
        for t, g in enumerate(gates):
            (key_g if t in oracle else fill_g).append(g)
    out = {
        "promote_key": float(np.mean(key_g)) if key_g else None,
        "promote_filler": float(np.mean(fill_g)) if fill_g else None,
    }
    if key_g and fill_g:
        out["promote_ratio"] = out["promote_key"] / max(out["promote_filler"], 1e-8)
    if hasattr(model, "summarize_diag"):
        extra = model.summarize_diag()
        out.update({k: v for k, v in extra.items() if v is not None})
    return out


def check_converged(curve, window=CONVERGE_WINDOW, tol_pp=CONVERGE_TOL_PP):
    if len(curve) < window:
        return False
    last = curve[-window:]
    for i in range(1, len(last)):
        if abs(last[i][1] - last[i - 1][1]) * 100 > tol_pp:
            return False
    return True


def run_one(config_name, seed, schedule_duration, ceiling, verbose=True):
    task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    model = make_model(config_name, seed)
    accs = []
    t0 = time.perf_counter()
    model.opt.zero_grad()
    curve = []
    diag_curve = []
    converged = False
    ep = 0
    while ep < ceiling:
        ep += 1
        model.opt.lr = lr_schedule(ep, schedule_duration)
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = model.forward_episode(episode, train=True)
        accs.append(acc)
        if ep % BATCH == 0:
            model.opt.step()
            model.opt.zero_grad()
        if ep % CHECKPOINT_EVERY == 0:
            ho = evaluate(model, n_episodes=150, seed=999)
            curve.append((ep, ho))
            d = collect_diagnostics(model)
            diag_curve.append({"ep": ep, **d})
            if verbose:
                ratio = d.get("promote_ratio")
                ratio_s = f"{ratio:.2f}x" if ratio is not None else "n/a"
                print(f"  [{config_name:12s} seed={seed}] ep {ep:6d}  "
                      f"lr={model.opt.lr:.5f}  "
                      f"train={np.mean(accs[-CHECKPOINT_EVERY:]):.3f}  "
                      f"held_out={ho:.3f}  "
                      f"prom_ratio={ratio_s}  "
                      f"({time.perf_counter()-t0:.1f}s)")
            if check_converged(curve):
                converged = True
                break
    return dict(
        config=config_name, seed=seed, curve=curve, diag_curve=diag_curve,
        converged=converged,
        final_acc=curve[-1][1] if curve else None,
        episodes_run=ep,
        final_diag=diag_curve[-1] if diag_curve else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                        choices=ALL_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--duration", type=int, default=9000)
    parser.add_argument("--ceiling", type=int, default=DEFAULT_CEILING)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="run_phase41_utility_results.json")
    args = parser.parse_args()

    if args.quick:
        args.seeds = args.seeds[:1]
        args.duration = 300
        args.ceiling = 300
        global CHECKPOINT_EVERY
        CHECKPOINT_EVERY = 100
        print("*** --quick smoke test ***\n")

    print(f"Configs: {args.configs}")
    print(f"Seeds:   {args.seeds}")
    print(f"Schedule: linear {LR_START} → {LR_END} over {args.duration}")
    print(f"Ceiling: {args.ceiling}  Convergence: {CONVERGE_WINDOW} ckpts within {CONVERGE_TOL_PP}pp\n")

    all_results = []
    for config_name in args.configs:
        print(f"\n{'='*70}\nConfig: {config_name}\n{'='*70}")
        for seed in args.seeds:
            r = run_one(config_name, seed, args.duration, args.ceiling)
            all_results.append(r)
            status = "CONVERGED" if r["converged"] else f"HIT CEILING {args.ceiling}"
            print(f"  -> seed={seed}  final_acc={r['final_acc']:.3f}  "
                  f"episodes={r['episodes_run']}  {status}")
            if r["final_diag"]:
                d = r["final_diag"]
                print(f"     promote_key={d.get('promote_key')}  "
                      f"promote_filler={d.get('promote_filler')}  "
                      f"ratio={d.get('promote_ratio')}")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'config':12s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}  {'conv':>6s}")
    summary = {}
    for config_name in args.configs:
        finals = np.array([r["final_acc"] for r in all_results if r["config"] == config_name])
        convs = [r["converged"] for r in all_results if r["config"] == config_name]
        print(f"{config_name:12s} {finals.mean():8.3f} {finals.std():8.3f} "
              f"{finals.min():8.3f} {finals.max():8.3f}  {sum(convs)}/{len(convs)}")
        summary[config_name] = dict(
            mean=float(finals.mean()), std=float(finals.std()),
            min=float(finals.min()), max=float(finals.max()),
            converged_count=int(sum(convs)), n_seeds=len(convs),
        )

    out = dict(args=vars(args), results=all_results, summary=summary)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
