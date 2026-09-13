"""
run_lr_schedule.py

The single most important experiment flagged as missing in
PHASE3_GATE_RESULTS.md's "Isolating the LR-decay effect from gate
supervision" section: the LR-decay treatment credited with AAC-diff's
biggest jump in this investigation had never been applied to the model
it's actually being compared against (AttentionBaseline), and every prior
LR-decay number in this project was a single-seed, fixed-schedule,
non-converged snapshot.

This script fixes all three gaps at once:

  1. A configurable LR schedule -- linear or cosine decay from lr_start
     to lr_end over a configurable duration, applied by mutating
     model.opt.lr every episode. Not hardcoded to one shape/duration:
     Phase 3 found results are sensitive to schedule duration, not just
     presence (a 6000-episode schedule and a 9000-episode schedule gave
     substantially different numbers for the identical seed/config), so
     both are exposed as CLI arguments rather than guessed at again.

  2. Applies it uniformly across all four configs that need a fair,
     matched comparison:
       - rnn            RNNBaseline           (no memory, floor)
       - attention      AttentionBaseline     (uncompressed KV -- the
                         baseline AAC-diff is measured against, and which
                         has never once been given this treatment)
       - aacdiff_noaux  AACDiffModel, decay=0.999, aux_gate_weight=0
       - aacdiff_aux    AACDiffModel, decay=0.999, aux_gate_weight=1,
                         aux_pos_weight=5 (the config identified in
                         run_aux_gate.py)

  3. Runs each (config, seed) to the SAME convergence criterion defined
     in the Robustness check: three consecutive 1000-episode held-out
     checkpoints within 2 percentage points of each other, up to a
     ceiling (default 15000 episodes). If a config never meets the
     criterion, that is reported explicitly -- the last checkpoint is
     never silently presented as a converged value.

  4. Repeats every config across multiple seeds (default 5, matching
     run_multiseed.py's precedent) and reports mean +/- std, min/max --
     never a single-seed point estimate.

Usage:
    # full spec (WARNING: this is a lot of compute -- see the runtime
    # estimate printed at startup before committing to a run)
    python3 run_lr_schedule.py

    # narrower runs while iterating
    python3 run_lr_schedule.py --configs attention aacdiff_aux --seeds 0 1 2
    python3 run_lr_schedule.py --schedule cosine --duration 6000
    python3 run_lr_schedule.py --ceiling 9000          # matches prior precedent, cheaper
    python3 run_lr_schedule.py --quick                 # tiny smoke test, NOT real results

Output:
    Prints the full checkpoint curve for every (config, seed) as it runs,
    a final mean+/-std summary table, and saves everything (every curve,
    every seed, convergence status) to --out (default
    run_lr_schedule_results.json) so the raw data survives independent of
    the printed log.
"""
import sys, time, argparse, json
import numpy as np

sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import RNNBaseline, AttentionBaseline, AACDiffModel, AdamTensor

D_EMB, D_H = 32, 32
N_VOCAB = 64
LR_START_DEFAULT = 1e-2
LR_END_DEFAULT = 1e-3
BATCH = 8
CHECKPOINT_EVERY = 1000
CONVERGE_WINDOW = 3        # consecutive checkpoints
CONVERGE_TOL_PP = 2.0      # percentage points
DEFAULT_CEILING = 15000
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
ALL_CONFIGS = ["rnn", "attention", "aacdiff_noaux", "aacdiff_aux"]

# rough per-episode wall-clock estimates from this investigation's own
# sandbox timings (d_emb=d_h=32, n_pairs=6, n_queries=6, default gap) --
# these are ballpark only; actual hardware will vary. Used purely to print
# a runtime estimate before a long run starts.
EST_MS_PER_EPISODE = {
    "rnn": 8.0,
    "attention": 10.0,
    "aacdiff_noaux": 18.0,
    "aacdiff_aux": 20.0,
}


def lr_schedule(ep, n_total, lr_start=LR_START_DEFAULT, lr_end=LR_END_DEFAULT, shape="linear"):
    """Learning rate for episode `ep` (1-indexed) of a schedule spanning
    `n_total` episodes. frac is clamped to [0,1], so training past
    n_total (e.g. because convergence hasn't been reached yet) holds at
    lr_end rather than extrapolating.

    shape="linear": lr_start + frac*(lr_end-lr_start)
    shape="cosine":  cosine anneal from lr_start to lr_end
    """
    frac = min(1.0, max(0.0, ep / n_total))
    if shape == "linear":
        return lr_start + frac * (lr_end - lr_start)
    elif shape == "cosine":
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + np.cos(np.pi * frac))
    else:
        raise ValueError(f"unknown schedule shape: {shape!r} (expected 'linear' or 'cosine')")


def make_model(config_name, seed):
    if config_name == "rnn":
        m = RNNBaseline(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed)
    elif config_name == "attention":
        m = AttentionBaseline(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed)
    elif config_name == "aacdiff_noaux":
        m = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed,
                          gate_mode="learned", decay=0.999,
                          aux_gate_weight=0.0, aux_pos_weight=1.0)
    elif config_name == "aacdiff_aux":
        m = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed,
                          gate_mode="learned", decay=0.999,
                          aux_gate_weight=1.0, aux_pos_weight=5.0)
    else:
        raise ValueError(f"unknown config: {config_name!r} (expected one of {ALL_CONFIGS})")
    m.opt = AdamTensor(m.params, lr=LR_START_DEFAULT)
    return m


def evaluate(model, n_episodes=150, gap_len=(20, 40), seed=999, n_vocab=N_VOCAB):
    eval_task = MQARTask(n_vocab=n_vocab, seed=seed)
    accs = []
    for _ in range(n_episodes):
        ep = eval_task.sample_episode(n_pairs=6, n_queries=6, gap_len=gap_len)
        _, acc = model.forward_episode(ep, train=False)
        accs.append(acc)
    return float(np.mean(accs))


def check_converged(curve, window=CONVERGE_WINDOW, tol_pp=CONVERGE_TOL_PP):
    """curve: list of (episode, acc) checkpoints in order. True iff every
    pair of adjacent checkpoints in the last `window` differs by at most
    tol_pp percentage points -- the exact criterion used in the
    Robustness check (e.g. "ep7000/8000/9000, deltas 3.1pp/2.9pp -> No")."""
    if len(curve) < window:
        return False
    last = curve[-window:]
    for i in range(1, len(last)):
        delta_pp = abs(last[i][1] - last[i - 1][1]) * 100
        if delta_pp > tol_pp:
            return False
    return True


def run_one(config_name, seed, schedule_shape, schedule_duration, ceiling,
            lr_start=LR_START_DEFAULT, lr_end=LR_END_DEFAULT, batch=BATCH,
            checkpoint_every=CHECKPOINT_EVERY, verbose=True):
    """Train one (config, seed) pair under the given LR schedule,
    checkpointing held-out accuracy every `checkpoint_every` episodes,
    stopping at the first of: convergence criterion met, or `ceiling`
    episodes reached. Returns the full checkpoint curve and convergence
    status -- never just a final number."""
    task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    model = make_model(config_name, seed)
    accs = []
    t0 = time.perf_counter()
    model.opt.zero_grad()
    curve = []
    converged = False
    ep = 0
    while ep < ceiling:
        ep += 1
        model.opt.lr = lr_schedule(ep, schedule_duration, lr_start, lr_end, schedule_shape)
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = model.forward_episode(episode, train=True)
        accs.append(acc)
        if ep % batch == 0:
            model.opt.step()
            model.opt.zero_grad()
        if ep % checkpoint_every == 0:
            ho = evaluate(model, n_episodes=150, seed=999)
            curve.append((ep, ho))
            if verbose:
                print(f"  [{config_name} seed={seed}] ep {ep:6d}  lr={model.opt.lr:.5f}  "
                      f"train_acc(last{checkpoint_every})={np.mean(accs[-checkpoint_every:]):.3f}  "
                      f"held_out={ho:.3f}  ({time.perf_counter()-t0:.1f}s)")
            if check_converged(curve):
                converged = True
                break
    return dict(config=config_name, seed=seed, curve=curve, converged=converged,
                final_acc=curve[-1][1] if curve else None, episodes_run=ep,
                schedule_shape=schedule_shape, schedule_duration=schedule_duration)


def estimate_runtime(configs, seeds, ceiling):
    total_s = 0.0
    for c in configs:
        ms = EST_MS_PER_EPISODE.get(c, 15.0)
        # very rough: assume convergence takes ~60% of ceiling on average;
        # this is a guess, not a guarantee -- purely to set expectations.
        assumed_episodes = ceiling * 0.6
        total_s += len(seeds) * assumed_episodes * ms / 1000.0
    return total_s


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS, choices=ALL_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--duration", type=int, default=9000,
                         help="episodes over which the LR decays from lr_start to lr_end")
    parser.add_argument("--lr-start", type=float, default=LR_START_DEFAULT)
    parser.add_argument("--lr-end", type=float, default=LR_END_DEFAULT)
    parser.add_argument("--ceiling", type=int, default=DEFAULT_CEILING,
                         help="hard stop if the convergence criterion is never met")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--quick", action="store_true",
                         help="smoke-test mode: tiny episode budget, NOT for real results")
    parser.add_argument("--out", default="run_lr_schedule_results.json",
                         help="where to save the full raw results (every curve, every seed)")
    args = parser.parse_args()

    if args.quick:
        args.seeds = args.seeds[:1]
        args.duration = 200
        args.ceiling = 200
        args.checkpoint_every = 50
        print("*** --quick mode: this is a smoke test, NOT a real result ***\n")

    est_s = estimate_runtime(args.configs, args.seeds, args.ceiling)
    print(f"Configs: {args.configs}")
    print(f"Seeds:   {args.seeds}")
    print(f"Schedule: {args.schedule}, {args.lr_start} -> {args.lr_end} over {args.duration} episodes")
    print(f"Convergence: {CONVERGE_WINDOW} consecutive {args.checkpoint_every}-ep checkpoints "
          f"within {CONVERGE_TOL_PP}pp, ceiling {args.ceiling}")
    print(f"Rough runtime estimate: ~{est_s/60:.0f} minutes total "
          f"(guess based on this investigation's own sandbox timings -- actual hardware varies, "
          f"and this assumes ~60% of ceiling to converge, which is not guaranteed)\n")

    all_results = []
    for config_name in args.configs:
        print(f"\n{'='*70}\nConfig: {config_name}\n{'='*70}")
        for seed in args.seeds:
            r = run_one(config_name, seed, args.schedule, args.duration, args.ceiling,
                        lr_start=args.lr_start, lr_end=args.lr_end, batch=args.batch,
                        checkpoint_every=args.checkpoint_every)
            all_results.append(r)
            status = "CONVERGED" if r["converged"] else f"NOT CONVERGED (hit ceiling {args.ceiling})"
            print(f"  -> seed={seed}  final_acc={r['final_acc']:.3f}  "
                  f"episodes_run={r['episodes_run']}  {status}")

    print(f"\n{'='*70}\nSUMMARY (mean +/- std across {len(args.seeds)} seeds)\n{'='*70}")
    print(f"{'config':16s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}  {'converged?':>12s}")
    summary = {}
    for config_name in args.configs:
        finals = np.array([r["final_acc"] for r in all_results if r["config"] == config_name])
        convs = [r["converged"] for r in all_results if r["config"] == config_name]
        conv_str = f"{sum(convs)}/{len(convs)}"
        print(f"{config_name:16s} {finals.mean():8.3f} {finals.std():8.3f} "
              f"{finals.min():8.3f} {finals.max():8.3f}  {conv_str:>12s}")
        summary[config_name] = dict(mean=float(finals.mean()), std=float(finals.std()),
                                     min=float(finals.min()), max=float(finals.max()),
                                     converged_count=int(sum(convs)), n_seeds=len(convs))
    if any(s["converged_count"] < s["n_seeds"] for s in summary.values()):
        print("\nNOTE: at least one config did not meet the convergence criterion for all "
              "seeds within the ceiling. Its 'final_acc' is a ceiling snapshot, not a "
              "converged value -- do not report it as final without saying so.")

    out = dict(args=vars(args), results=all_results, summary=summary)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results (every curve, every seed) saved to {args.out}")


if __name__ == "__main__":
    main()