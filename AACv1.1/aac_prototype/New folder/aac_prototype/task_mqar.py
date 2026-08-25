"""
Standard "induction head" / multi-query associative-recall (MQAR) task,
the format used in the SSM-vs-attention literature (e.g. Based, H3,
Zoology) to test long-range selective recall. Unlike task.py (the earlier
prototype's task), there is NO separate KEY/VALUE/QUERY vocabulary and NO
privileged shared identity embedding between a key and its later query --
every token is drawn from one shared vocabulary, and a "query" is just an
earlier key token literally reappearing.

Sequence construction:
  - n_pairs distinct (key, value) tokens, both drawn from the same shared
    vocabulary of size n_vocab, key != value, keys distinct from each
    other
  - pairs separated by filler tokens (also drawn from the same shared
    vocabulary, so filler is not a separate namespace either)
  - after a long gap, each key token reappears bare; the model must
    predict, at that position, the value token that followed it the
    first time

This is strictly harder than task.py: nothing hands the model a
pre-aligned representation for "this is a key" vs. "this is a query" --
if a model wants to solve this, it has to learn, purely from training
signal, to (a) recognize when a token is worth remembering, (b) bind it
to whatever follows, and (c) recognize the same token on repeat and
retrieve the binding -- using representations it has to learn from
scratch.
"""
import numpy as np


class MQARTask:
    def __init__(self, n_vocab=64, seed=0):
        self.n_vocab = n_vocab
        self.vocab_size = n_vocab  # single shared vocabulary
        self.rng = np.random.default_rng(seed)

    def sample_episode(self, n_pairs=6, filler_between=(2, 6), gap_len=(20, 40),
                        n_queries=6):
        rng = self.rng
        # keys and values are just distinct tokens from the shared vocab
        chosen = rng.choice(self.n_vocab, size=2 * n_pairs, replace=False)
        keys, vals = chosen[:n_pairs], chosen[n_pairs:]
        val_of = {int(k): int(v) for k, v in zip(keys, vals)}

        seq, targets = [], {}  # targets: position -> label (only at query positions)
        key_positions = []  # diagnostic only, never used in the loss

        def emit_filler(n):
            for _ in range(n):
                seq.append(int(rng.integers(0, self.n_vocab)))

        for k in keys:
            key_positions.append(len(seq))
            seq.append(int(k))
            seq.append(val_of[int(k)])
            emit_filler(int(rng.integers(*filler_between)))

        emit_filler(int(rng.integers(*gap_len)))

        query_keys = rng.choice(keys, size=n_queries, replace=True)
        for k in query_keys:
            seq.append(int(k))
            targets[len(seq) - 1] = val_of[int(k)]
            emit_filler(int(rng.integers(1, 4)))

        return {
            "tokens": np.array(seq, dtype=int),
            "query_positions": list(targets.keys()),
            "query_labels": list(targets.values()),
            "key_positions": key_positions,  # diagnostic only
        }
