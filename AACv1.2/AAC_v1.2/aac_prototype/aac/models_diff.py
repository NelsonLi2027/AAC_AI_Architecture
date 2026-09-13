import numpy as np
from aac import autodiff as ad


class AdamTensor:
    def __init__(self, params, lr=5e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, p in self.params.items():
            g = p.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for p in self.params.values():
            p.zero_grad()


def _init(rng, shape, fan_in=None):
    fan_in = fan_in or (shape[-1] if len(shape) > 1 else shape[0])
    return ad.Tensor(rng.normal(0, 1.0 / np.sqrt(fan_in), size=shape),
                      requires_grad=True)


def _zeros(shape):
    return ad.Tensor(np.zeros(shape), requires_grad=True)


class RNNBaseline:
    name = "RNN-only (no memory)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Whh": _init(rng, (d_h, d_h)),
            "Wxh": _init(rng, (d_h, d_emb)),
            "bh": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        self.opt = AdamTensor(self.params)
        self.d_h = d_h

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        h = ad.Tensor(np.zeros(self.d_h))
        losses, correct, total = [], 0, 0

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            h = ad.tanh(ad.add(ad.add(ad.matvec(p["Whh"], h),
                                       ad.matvec(p["Wxh"], emb)), p["bh"]))
            if t in qpos:
                logits = ad.add(ad.matvec(p["Wout"], h), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

        return self._finish(losses, correct, total, train)

    def _finish(self, losses, correct, total, train):
        if not losses:
            return 0.0, 0.0
        total_loss = losses[0]
        for l in losses[1:]:
            total_loss = ad.add(total_loss, l)
        if train:
            total_loss.backward()
        else:
            # eval mode never calls .backward(), so the cycle-breaking
            # that lives there never runs -- do it explicitly here instead,
            # or every eval call leaks its whole graph (see autodiff.py's
            # free_graph docstring for the full explanation).
            ad.free_graph(total_loss)
        return float(total_loss.data) / total, correct / total


class AttentionBaseline:
    """One-layer causal self-attention over the full token history. This
    model's memory is UNCOMPRESSED -- it never forgets or throws anything
    away, so its capacity grows with sequence length. It's a ceiling on
    achievable accuracy, not a fair capacity match against AACDiffModel's
    fixed-size memory (see PHASE2_RESULTS.md)."""
    name = "Single-layer causal self-attention (uncompressed KV)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Wq": _init(rng, (d_h, d_emb)),
            "Wk": _init(rng, (d_h, d_emb)),
            "Wv": _init(rng, (d_h, d_emb)),
            "Wcomb": _init(rng, (d_h, d_emb + d_h)),
            "bcomb": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        self.opt = AdamTensor(self.params)
        self.d_h = d_h

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        K_hist, V_hist = [], []
        losses, correct, total = [], 0, 0
        scale = 1.0 / np.sqrt(self.d_h)
        prev_emb = None

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            if prev_emb is not None:
                # Bind (previous token as key) -> (current token as value):
                # this is what makes "attend back to an earlier occurrence
                # of this token" actually recover the token that FOLLOWED
                # it, rather than the key's own embedding again (see the
                # induction-head binding-shift bug fix in PHASE2_RESULTS.md).
                k = ad.matvec(p["Wk"], prev_emb)
                v = ad.matvec(p["Wv"], emb)
                K_hist.append(k)
                V_hist.append(v)

            if t in qpos and K_hist:
                q = ad.matvec(p["Wq"], emb)
                K_stack = ad.stack(K_hist)
                V_stack = ad.stack(V_hist)
                scores = ad.mul_const(ad.matvec(K_stack, q), scale)
                weights = ad.softmax(scores)
                read = ad.vecmat(weights, V_stack)
                comb = ad.concat(emb, read)
                hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
                logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

            prev_emb = emb

        return RNNBaseline._finish(self, losses, correct, total, train)


class AACDiffModel:
    name = "AAC-diff (fast state + gated fixed-size associative memory)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, decay=0.95, beta=0.1,
                 update_rule="hebbian", gate_mode="learned", gate_bias_init=0.0,
                 aux_gate_weight=0.0, aux_pos_weight=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Whh": _init(rng, (d_h, d_h)),
            "Wxh": _init(rng, (d_h, d_emb)),
            "bh": _zeros((d_h,)),
            "Wk": _init(rng, (d_h, d_emb)),
            "Wv": _init(rng, (d_h, d_emb)),
            "Wq": _init(rng, (d_h, d_emb)),
            "Wg": _init(rng, (1, d_h)),
            # gate bias: >0 warm-starts the gate toward "open" (sigmoid(bg))
            "bg": ad.Tensor(np.array([gate_bias_init]), requires_grad=True),
            "Wcomb": _init(rng, (d_h, 2 * d_h)),
            "bcomb": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        self.opt = AdamTensor(self.params)
        self.d_h = d_h
        self.decay = decay
        self.beta = beta
        self.update_rule = update_rule
        self.gate_mode = gate_mode  # "learned", "oracle", "oracle_decay", "curriculum"
        self.alpha = 1.0  # curriculum mixing weight: 0 = pure oracle write,
        # 1 = pure learned-gate write. Gate is ALWAYS computed & trained via
        # backprop in curriculum mode; alpha only controls how much of the
        # actual M update comes from the learned gate vs. the oracle mask.
        # Auxiliary supervised loss on the gate itself: BCE(g_t, oracle_label_t)
        # added to the main task loss, weighted by aux_gate_weight (0 = off,
        # i.e. exactly the old unsupervised "learned" mode). aux_pos_weight
        # upweights the rare positive (key-write) class against the far more
        # common filler class.
        self.aux_gate_weight = aux_gate_weight
        self.aux_pos_weight = aux_pos_weight
        self.last_gate_values = []
        self.grad_norms = {"Wg": [], "Wk": [], "Wv": [], "Wq": []}

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        oracle_positions = set(int(pos) for pos in
                                [kp + 1 for kp in episode["key_positions"] if kp + 1 < len(tokens)])
        h = ad.Tensor(np.zeros(self.d_h))
        M = ad.Tensor(np.zeros((self.d_h, self.d_h)))
        losses, correct, total = [], 0, 0
        aux_losses = []
        self.last_gate_values = []
        prev_emb = None

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            h = ad.tanh(ad.add(ad.add(ad.matvec(p["Whh"], h),
                                       ad.matvec(p["Wxh"], emb)), p["bh"]))

            if prev_emb is not None:
                k = ad.matvec(p["Wk"], prev_emb)
                v = ad.matvec(p["Wv"], emb)
                if self.gate_mode == "oracle":
                    # oracle: hard 1/0 write at exactly the correct positions,
                    # NOT a function of any trainable param -> no gradient
                    # flows to a "gate", by construction. Wg/bg are unused.
                    # NOTE: no decay applied here (accumulates undamped).
                    if t in oracle_positions:
                        M = ad.add(M, ad.outer(k, v))
                    self.last_gate_values.append(1.0 if t in oracle_positions else 0.0)

                elif self.gate_mode == "oracle_decay":
                    # controlled variant: SAME hard oracle write timing as
                    # "oracle", but WITH the same decay=0.95 applied every
                    # step as the learned/curriculum paths use. Isolates
                    # "correct write timing" from "no decay at all".
                    oracle_val = 1.0 if t in oracle_positions else 0.0
                    M = ad.add(ad.mul_const(M, self.decay),
                               ad.mul_const(ad.outer(k, v), oracle_val))
                    self.last_gate_values.append(oracle_val)

                elif self.gate_mode == "curriculum":
                    # gate is ALWAYS computed and trained (real gradient to
                    # Wg/bg every step), but the write actually applied to M
                    # is a blend of the learned gate and the oracle mask.
                    # alpha=0 -> pure oracle write (like oracle mode) but
                    # gate is still being trained on the side; alpha=1 ->
                    # pure learned-gate write (like learned mode). Ramp
                    # alpha 0->1 across training (set externally per-episode).
                    g = ad.sigmoid(ad.add(ad.matvec(p["Wg"], h), p["bg"]))
                    self.last_gate_values.append(float(g.data[0]))
                    oracle_val = 1.0 if t in oracle_positions else 0.0
                    # g_eff = alpha*g + (1-alpha)*oracle_val  (oracle_val is a
                    # python float constant, not a Tensor -> no extra graph)
                    g_eff_data = self.alpha * g.data + (1 - self.alpha) * oracle_val
                    g_eff = ad.Tensor(g_eff_data, _children=(g,), _op="curric_blend")

                    def _mk_backward(gate_t, alpha_local, out_t):
                        # bind gate_t/alpha_local/out_t as defaults so each
                        # timestep's closure captures ITS OWN tensors, not
                        # whatever the loop variables point to by the time
                        # .backward() actually runs (late-binding bug fixed).
                        def _b():
                            if gate_t.requires_grad:
                                gate_t.grad += out_t.grad * alpha_local
                        return _b
                    g_eff._backward = _mk_backward(g, self.alpha, g_eff)

                    M = ad.add(ad.mul_const(M, self.decay), ad.scale(ad.outer(k, v), g_eff))

                else:  # "learned" (with optional warm-start bias bg)
                    g = ad.sigmoid(ad.add(ad.matvec(p["Wg"], h), p["bg"]))
                    self.last_gate_values.append(float(g.data[0]))
                    if self.aux_gate_weight > 0.0:
                        oracle_val = 1.0 if t in oracle_positions else 0.0
                        aux_losses.append(ad.mul_const(
                            ad.bce_loss(g, oracle_val, pos_weight=self.aux_pos_weight),
                            self.aux_gate_weight))
                    M = ad.add(ad.mul_const(M, self.decay), ad.scale(ad.outer(k, v), g))
            else:
                self.last_gate_values.append(0.0)

            if t in qpos:
                q = ad.matvec(p["Wq"], emb)
                read = ad.vecmat(q, M)
                comb = ad.concat(h, read)
                hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
                logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

            prev_emb = emb

        result = self._finish_with_aux(losses, aux_losses, correct, total, train)
        if train and self.opt.t <= 500 and self.opt.t % 50 == 0:
            for key in ["Wg", "Wk", "Wv", "Wq"]:
                if key in p:
                    gnorm = float(np.linalg.norm(p[key].grad))
                    self.grad_norms[key].append((self.opt.t, gnorm))
        return result

    def _finish_with_aux(self, losses, aux_losses, correct, total, train):
        """Same as RNNBaseline._finish, but adds the auxiliary gate-BCE
        terms into the SAME backward pass (so Wg/bg get gradient from both
        the task loss and the direct supervision), while still reporting
        only the task loss/accuracy for comparability with the unsupervised
        runs."""
        if not losses:
            return 0.0, 0.0
        total_loss = losses[0]
        for l in losses[1:]:
            total_loss = ad.add(total_loss, l)
        task_loss_val = float(total_loss.data)
        combined = total_loss
        for al in aux_losses:
            combined = ad.add(combined, al)
        if train:
            combined.backward()
        else:
            # same fix as RNNBaseline._finish -- eval-mode never calls
            # .backward(), so break the cycles explicitly here instead.
            ad.free_graph(combined)
        return task_loss_val / total, correct / total


class AACDiffUtility:
    """Phase 4.1: learned utility-based write/promotion.

    Architectural additions over AACDiffModel:
      - Explicit Temporary Trace T (vector evidence accumulator).
      - Value estimator Value_θ(e, S, T) → scalar predicted future value.
      - Soft promotion probability p_promote = sigmoid(v) used as the
        differentiable write gate into Persistent Memory P.
      - Training signal is *only* the downstream task loss. No oracle
        write labels are used as a training target (oracle positions are
        retained solely for diagnostic logging of promotion rates).

    Decision vocabulary supported by the controller:
      IGNORE          – low keep-in-trace and low promote
      KEEP IN TRACE   – high keep-in-trace, low promote
      PROMOTE         – high promote (T evidence flows into P)

    For the initial Phase 4.1 implementation the keep-in-trace gate is
    a learned scalar as well; both gates are trained end-to-end from the
    task objective.
    """
    name = "AAC-diff utility (S + T + learned Value → promote → P)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, decay_p=0.999,
                 decay_t=0.9, value_hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            # shared backbone
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Whh": _init(rng, (d_h, d_h)),
            "Wxh": _init(rng, (d_h, d_emb)),
            "bh": _zeros((d_h,)),
            # memory key/value projections
            "Wk": _init(rng, (d_h, d_emb)),
            "Wv": _init(rng, (d_h, d_emb)),
            "Wq": _init(rng, (d_h, d_emb)),
            # Temporary Trace write (keep-in-trace gate)
            "W_keep": _init(rng, (1, d_h + d_emb)),
            "b_keep": ad.Tensor(np.array([0.0]), requires_grad=True),
            # Value estimator: predicts future utility of retaining the
            # current candidate. Input = concat(S, T, emb).
            "Wv1": _init(rng, (value_hidden, 2 * d_h + d_emb)),
            "bv1": _zeros((value_hidden,)),
            "Wv2": _init(rng, (1, value_hidden)),
            "bv2": ad.Tensor(np.array([0.0]), requires_grad=True),
            # readout
            "Wcomb": _init(rng, (d_h, 2 * d_h)),
            "bcomb": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        self.opt = AdamTensor(self.params)
        self.d_h = d_h
        self.d_emb = d_emb
        self.decay_p = decay_p          # fixed persistence for P (Phase 4.2 will learn it)
        self.decay_t = decay_t          # fixed decay for T
        # diagnostics collected during the most recent forward
        self.last_gate_values = []      # p_promote per step (compat with old diagnostics)
        self.last_keep_values = []
        self.last_value_scores = []
        self.diag = {
            "promote_key": [],
            "promote_filler": [],
            "keep_key": [],
            "keep_filler": [],
            "value_key": [],
            "value_filler": [],
            "trace_norm": [],
            "mem_norm": [],
        }

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        # oracle positions are used *only* for diagnostic logging, never
        # as a training target.
        oracle_write = set(
            int(pos) for pos in
            [kp + 1 for kp in episode["key_positions"] if kp + 1 < len(tokens)]
        )

        S = ad.Tensor(np.zeros(self.d_h))          # fast state
        T = ad.Tensor(np.zeros(self.d_h))          # temporary trace
        P = ad.Tensor(np.zeros((self.d_h, self.d_h)))  # persistent associative memory

        losses, correct, total = [], 0, 0
        self.last_gate_values = []
        self.last_keep_values = []
        self.last_value_scores = []
        prev_emb = None

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            # Fast state update
            S = ad.tanh(ad.add(ad.add(ad.matvec(p["Whh"], S),
                                       ad.matvec(p["Wxh"], emb)), p["bh"]))

            if prev_emb is not None:
                k = ad.matvec(p["Wk"], prev_emb)
                v = ad.matvec(p["Wv"], emb)

                # --- Temporary Trace write (KEEP IN TRACE) ---
                # soft gate from (S, emb)
                keep_in = ad.concat(S, emb)
                g_keep = ad.sigmoid(ad.add(ad.matvec(p["W_keep"], keep_in), p["b_keep"]))
                # T accumulates soft evidence
                T = ad.add(ad.mul_const(T, self.decay_t),
                           ad.scale(v, g_keep))   # write value embedding scaled by keep gate

                # --- Value estimator → promotion probability ---
                # Input: concat(S, T, emb)
                val_in = ad.concat(ad.concat(S, T), emb)
                h_v = ad.tanh(ad.add(ad.matvec(p["Wv1"], val_in), p["bv1"]))
                v_score = ad.add(ad.matvec(p["Wv2"], h_v), p["bv2"])
                p_promote = ad.sigmoid(v_score)

                # --- Persistent Memory update (PROMOTE) ---
                P = ad.add(ad.mul_const(P, self.decay_p),
                           ad.scale(ad.outer(k, v), p_promote))

                # logging (numpy side only)
                g_keep_f = float(g_keep.data[0])
                p_prom_f = float(p_promote.data[0])
                v_sc_f = float(v_score.data[0])
                self.last_keep_values.append(g_keep_f)
                self.last_gate_values.append(p_prom_f)
                self.last_value_scores.append(v_sc_f)

                is_key = t in oracle_write
                if is_key:
                    self.diag["promote_key"].append(p_prom_f)
                    self.diag["keep_key"].append(g_keep_f)
                    self.diag["value_key"].append(v_sc_f)
                else:
                    self.diag["promote_filler"].append(p_prom_f)
                    self.diag["keep_filler"].append(g_keep_f)
                    self.diag["value_filler"].append(v_sc_f)
            else:
                self.last_keep_values.append(0.0)
                self.last_gate_values.append(0.0)
                self.last_value_scores.append(0.0)

            # --- Associative read + prediction at query positions ---
            if t in qpos:
                q = ad.matvec(p["Wq"], emb)
                read = ad.vecmat(q, P)
                comb = ad.concat(S, read)
                hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
                logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

            prev_emb = emb

        # final diagnostics for this episode
        self.diag["trace_norm"].append(float(np.linalg.norm(T.data)))
        self.diag["mem_norm"].append(float(np.linalg.norm(P.data)))

        return self._finish(losses, correct, total, train)

    def _finish(self, losses, correct, total, train):
        if not losses:
            return 0.0, 0.0
        total_loss = losses[0]
        for l in losses[1:]:
            total_loss = ad.add(total_loss, l)
        if train:
            total_loss.backward()
        else:
            ad.free_graph(total_loss)
        return float(total_loss.data) / total, correct / total

    def reset_diag(self):
        for k in self.diag:
            self.diag[k] = []

    def summarize_diag(self):
        """Return mean promotion / keep / value rates split by key vs filler.
        Safe to call after one or more forward passes; returns None for empty."""
        def _m(xs):
            return float(np.mean(xs)) if xs else None
        return {
            "promote_key": _m(self.diag["promote_key"]),
            "promote_filler": _m(self.diag["promote_filler"]),
            "keep_key": _m(self.diag["keep_key"]),
            "keep_filler": _m(self.diag["keep_filler"]),
            "value_key": _m(self.diag["value_key"]),
            "value_filler": _m(self.diag["value_filler"]),
            "trace_norm": _m(self.diag["trace_norm"]),
            "mem_norm": _m(self.diag["mem_norm"]),
            "promote_ratio": (
                _m(self.diag["promote_key"]) / max(_m(self.diag["promote_filler"]), 1e-8)
                if self.diag["promote_key"] and self.diag["promote_filler"] else None
            ),
        }


class AACDiffPersist:
    """Phase 4.2: learned / state-dependent persistence on top of utility promotion.

    Persistence modes (controlled ablation):
      "fixed"  — λ_p = decay_p constant (baseline from Phase 4.1)
      "global" — single learnable scalar λ = sigmoid(b_λ), shared across time
      "state"  — λ_t = sigmoid(f_θ(S_t, T_t, emb))  state-dependent per step

    Temporary-trace decay remains fixed (decay_t) in this phase so the
    persistence ablation is isolated to P. Promotion is the same learned
    Value_θ path as AACDiffUtility (no oracle training target).
    """
    name = "AAC-diff persist (S + T + Value → promote + learned λ → P)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, decay_p=0.999,
                 decay_t=0.9, value_hidden=16, persist_mode="fixed",
                 persist_hidden=8, seed=0):
        if persist_mode not in ("fixed", "global", "state"):
            raise ValueError(f"persist_mode must be fixed|global|state, got {persist_mode!r}")
        rng = np.random.default_rng(seed)
        self.params = {
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Whh": _init(rng, (d_h, d_h)),
            "Wxh": _init(rng, (d_h, d_emb)),
            "bh": _zeros((d_h,)),
            "Wk": _init(rng, (d_h, d_emb)),
            "Wv": _init(rng, (d_h, d_emb)),
            "Wq": _init(rng, (d_h, d_emb)),
            # keep-in-trace gate
            "W_keep": _init(rng, (1, d_h + d_emb)),
            "b_keep": ad.Tensor(np.array([0.0]), requires_grad=True),
            # Value estimator
            "Wv1": _init(rng, (value_hidden, 2 * d_h + d_emb)),
            "bv1": _zeros((value_hidden,)),
            "Wv2": _init(rng, (1, value_hidden)),
            "bv2": ad.Tensor(np.array([0.0]), requires_grad=True),
            # readout
            "Wcomb": _init(rng, (d_h, 2 * d_h)),
            "bcomb": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        # persistence parameters (mode-dependent)
        if persist_mode == "global":
            # learnable bias; sigmoid(b_λ) starts near decay_p via logit
            # clip for numerical safety
            p0 = float(np.clip(decay_p, 1e-4, 1 - 1e-4))
            b0 = np.log(p0 / (1 - p0))
            self.params["b_lambda"] = ad.Tensor(np.array([b0]), requires_grad=True)
        elif persist_mode == "state":
            # small MLP: concat(S, T, emb) → λ_t
            self.params["Wl1"] = _init(rng, (persist_hidden, 2 * d_h + d_emb))
            self.params["bl1"] = _zeros((persist_hidden,))
            self.params["Wl2"] = _init(rng, (1, persist_hidden))
            # bias toward high persistence initially
            p0 = float(np.clip(decay_p, 1e-4, 1 - 1e-4))
            b0 = np.log(p0 / (1 - p0))
            self.params["bl2"] = ad.Tensor(np.array([b0]), requires_grad=True)

        self.opt = AdamTensor(self.params)
        self.d_h = d_h
        self.d_emb = d_emb
        self.decay_p = decay_p
        self.decay_t = decay_t
        self.persist_mode = persist_mode

        self.last_gate_values = []
        self.last_keep_values = []
        self.last_value_scores = []
        self.last_lambda_values = []
        self.diag = {
            "promote_key": [], "promote_filler": [],
            "keep_key": [], "keep_filler": [],
            "value_key": [], "value_filler": [],
            "lambda_key": [], "lambda_filler": [],
            "lambda_all": [],
            "trace_norm": [], "mem_norm": [],
        }

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        oracle_write = set(
            int(pos) for pos in
            [kp + 1 for kp in episode["key_positions"] if kp + 1 < len(tokens)]
        )

        S = ad.Tensor(np.zeros(self.d_h))
        T = ad.Tensor(np.zeros(self.d_h))
        P = ad.Tensor(np.zeros((self.d_h, self.d_h)))

        losses, correct, total = [], 0, 0
        self.last_gate_values = []
        self.last_keep_values = []
        self.last_value_scores = []
        self.last_lambda_values = []
        prev_emb = None

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            S = ad.tanh(ad.add(ad.add(ad.matvec(p["Whh"], S),
                                       ad.matvec(p["Wxh"], emb)), p["bh"]))

            if prev_emb is not None:
                k = ad.matvec(p["Wk"], prev_emb)
                v = ad.matvec(p["Wv"], emb)

                # keep-in-trace
                keep_in = ad.concat(S, emb)
                g_keep = ad.sigmoid(ad.add(ad.matvec(p["W_keep"], keep_in), p["b_keep"]))
                T = ad.add(ad.mul_const(T, self.decay_t), ad.scale(v, g_keep))

                # value → promote
                val_in = ad.concat(ad.concat(S, T), emb)
                h_v = ad.tanh(ad.add(ad.matvec(p["Wv1"], val_in), p["bv1"]))
                v_score = ad.add(ad.matvec(p["Wv2"], h_v), p["bv2"])
                p_promote = ad.sigmoid(v_score)

                # --- persistence λ ---
                # ad.scale works for any array shape (elementwise * scalar), so it
                # is valid for both the matrix P and the outer(k,v) write.
                if self.persist_mode == "fixed":
                    P = ad.add(ad.mul_const(P, self.decay_p),
                               ad.scale(ad.outer(k, v), p_promote))
                    lam_f = self.decay_p
                elif self.persist_mode == "global":
                    lam = ad.sigmoid(p["b_lambda"])
                    P = ad.add(ad.scale(P, lam),
                               ad.scale(ad.outer(k, v), p_promote))
                    lam_f = float(lam.data[0])
                else:  # state-dependent
                    hin = ad.concat(ad.concat(S, T), emb)
                    h_l = ad.tanh(ad.add(ad.matvec(p["Wl1"], hin), p["bl1"]))
                    lam = ad.sigmoid(ad.add(ad.matvec(p["Wl2"], h_l), p["bl2"]))
                    P = ad.add(ad.scale(P, lam),
                               ad.scale(ad.outer(k, v), p_promote))
                    lam_f = float(lam.data[0])

                # logging
                g_keep_f = float(g_keep.data[0])
                p_prom_f = float(p_promote.data[0])
                v_sc_f = float(v_score.data[0])
                self.last_keep_values.append(g_keep_f)
                self.last_gate_values.append(p_prom_f)
                self.last_value_scores.append(v_sc_f)
                self.last_lambda_values.append(lam_f)

                is_key = t in oracle_write
                if is_key:
                    self.diag["promote_key"].append(p_prom_f)
                    self.diag["keep_key"].append(g_keep_f)
                    self.diag["value_key"].append(v_sc_f)
                    self.diag["lambda_key"].append(lam_f)
                else:
                    self.diag["promote_filler"].append(p_prom_f)
                    self.diag["keep_filler"].append(g_keep_f)
                    self.diag["value_filler"].append(v_sc_f)
                    self.diag["lambda_filler"].append(lam_f)
                self.diag["lambda_all"].append(lam_f)
            else:
                self.last_keep_values.append(0.0)
                self.last_gate_values.append(0.0)
                self.last_value_scores.append(0.0)
                self.last_lambda_values.append(self.decay_p)

            if t in qpos:
                q = ad.matvec(p["Wq"], emb)
                read = ad.vecmat(q, P)
                comb = ad.concat(S, read)
                hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
                logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

            prev_emb = emb

        self.diag["trace_norm"].append(float(np.linalg.norm(T.data)))
        self.diag["mem_norm"].append(float(np.linalg.norm(P.data)))
        return self._finish(losses, correct, total, train)

    def _finish(self, losses, correct, total, train):
        if not losses:
            return 0.0, 0.0
        total_loss = losses[0]
        for l in losses[1:]:
            total_loss = ad.add(total_loss, l)
        if train:
            total_loss.backward()
        else:
            ad.free_graph(total_loss)
        return float(total_loss.data) / total, correct / total

    def reset_diag(self):
        for k in self.diag:
            self.diag[k] = []

    def summarize_diag(self):
        def _m(xs):
            return float(np.mean(xs)) if xs else None
        out = {
            "promote_key": _m(self.diag["promote_key"]),
            "promote_filler": _m(self.diag["promote_filler"]),
            "keep_key": _m(self.diag["keep_key"]),
            "keep_filler": _m(self.diag["keep_filler"]),
            "value_key": _m(self.diag["value_key"]),
            "value_filler": _m(self.diag["value_filler"]),
            "lambda_mean": _m(self.diag["lambda_all"]),
            "lambda_key": _m(self.diag["lambda_key"]),
            "lambda_filler": _m(self.diag["lambda_filler"]),
            "trace_norm": _m(self.diag["trace_norm"]),
            "mem_norm": _m(self.diag["mem_norm"]),
        }
        if self.diag["promote_key"] and self.diag["promote_filler"]:
            out["promote_ratio"] = (
                out["promote_key"] / max(out["promote_filler"], 1e-8)
            )
        return out


class AACDiffPipeline:
    """Phase 4.3: genuine differentiable Temporary Trace → Persistent Memory.

    Architectural distinction enforced:
      T ≠ P
      T accumulates candidate evidence over time.
      Promotion writes *from T* into P (not only the instantaneous outer(k,v)).

    Ablation modes (`pipeline_mode`):
      "direct" — write outer(k,v) * p_promote into P (Phase 4.1/4.2 style).
                 T is only a feature for the Value head, not the write source.
      "trace"  — T stores soft key/value candidates; promotion copies a
                 transform of T into P. This is the true T→P pipeline.

    Persistence on P defaults to learned global λ (winner of Phase 4.2).
    Temporary-trace decay remains fixed (decay_t) unless overridden.
    """
    name = "AAC-diff pipeline (S + T → promote → P)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, decay_p=0.999,
                 decay_t=0.9, value_hidden=16, pipeline_mode="trace",
                 persist_mode="global", seed=0):
        if pipeline_mode not in ("direct", "trace"):
            raise ValueError(f"pipeline_mode must be direct|trace, got {pipeline_mode!r}")
        if persist_mode not in ("fixed", "global"):
            # keep state out of this phase to isolate the T→P question
            raise ValueError(f"persist_mode must be fixed|global for pipeline, got {persist_mode!r}")
        rng = np.random.default_rng(seed)
        self.params = {
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Whh": _init(rng, (d_h, d_h)),
            "Wxh": _init(rng, (d_h, d_emb)),
            "bh": _zeros((d_h,)),
            "Wk": _init(rng, (d_h, d_emb)),
            "Wv": _init(rng, (d_h, d_emb)),
            "Wq": _init(rng, (d_h, d_emb)),
            # keep-in-trace gate (how much of current event enters T)
            "W_keep": _init(rng, (1, d_h + d_emb)),
            "b_keep": ad.Tensor(np.array([0.0]), requires_grad=True),
            # Value estimator → promotion probability
            "Wv1": _init(rng, (value_hidden, 2 * d_h + d_emb)),
            "bv1": _zeros((value_hidden,)),
            "Wv2": _init(rng, (1, value_hidden)),
            "bv2": ad.Tensor(np.array([0.0]), requires_grad=True),
            # readout
            "Wcomb": _init(rng, (d_h, 2 * d_h)),
            "bcomb": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        if pipeline_mode == "trace":
            # T stores separate key-side and value-side content vectors.
            # Promotion writes outer(T_k, T_v) (or a learned projection) into P.
            # Optional projection so T content can be remapped at promote time.
            self.params["W_tk"] = _init(rng, (d_h, d_h))
            self.params["W_tv"] = _init(rng, (d_h, d_h))
        if persist_mode == "global":
            p0 = float(np.clip(decay_p, 1e-4, 1 - 1e-4))
            b0 = np.log(p0 / (1 - p0))
            self.params["b_lambda"] = ad.Tensor(np.array([b0]), requires_grad=True)

        self.opt = AdamTensor(self.params)
        self.d_h = d_h
        self.d_emb = d_emb
        self.decay_p = decay_p
        self.decay_t = decay_t
        self.pipeline_mode = pipeline_mode
        self.persist_mode = persist_mode

        self.last_gate_values = []
        self.last_keep_values = []
        self.last_lambda_values = []
        self.diag = {
            "promote_key": [], "promote_filler": [],
            "keep_key": [], "keep_filler": [],
            "lambda_all": [],
            "trace_norm": [], "mem_norm": [],
            "tk_norm": [], "tv_norm": [],
        }

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        oracle_write = set(
            int(pos) for pos in
            [kp + 1 for kp in episode["key_positions"] if kp + 1 < len(tokens)]
        )

        S = ad.Tensor(np.zeros(self.d_h))
        # Temporary trace: content buffers (and a summary vector for Value)
        T_sum = ad.Tensor(np.zeros(self.d_h))   # summary evidence for Value head
        T_k = ad.Tensor(np.zeros(self.d_h))     # key-side candidate content
        T_v = ad.Tensor(np.zeros(self.d_h))     # value-side candidate content
        P = ad.Tensor(np.zeros((self.d_h, self.d_h)))

        losses, correct, total = [], 0, 0
        self.last_gate_values = []
        self.last_keep_values = []
        self.last_lambda_values = []
        prev_emb = None

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            S = ad.tanh(ad.add(ad.add(ad.matvec(p["Whh"], S),
                                       ad.matvec(p["Wxh"], emb)), p["bh"]))

            if prev_emb is not None:
                k = ad.matvec(p["Wk"], prev_emb)
                v = ad.matvec(p["Wv"], emb)

                # --- write into Temporary Trace (KEEP IN TRACE) ---
                keep_in = ad.concat(S, emb)
                g_keep = ad.sigmoid(ad.add(ad.matvec(p["W_keep"], keep_in), p["b_keep"]))
                T_sum = ad.add(ad.mul_const(T_sum, self.decay_t), ad.scale(v, g_keep))
                if self.pipeline_mode == "trace":
                    T_k = ad.add(ad.mul_const(T_k, self.decay_t), ad.scale(k, g_keep))
                    T_v = ad.add(ad.mul_const(T_v, self.decay_t), ad.scale(v, g_keep))

                # --- Value → promotion probability ---
                val_in = ad.concat(ad.concat(S, T_sum), emb)
                h_v = ad.tanh(ad.add(ad.matvec(p["Wv1"], val_in), p["bv1"]))
                v_score = ad.add(ad.matvec(p["Wv2"], h_v), p["bv2"])
                p_promote = ad.sigmoid(v_score)

                # --- persistence λ on P ---
                if self.persist_mode == "fixed":
                    P_decayed = ad.mul_const(P, self.decay_p)
                    lam_f = self.decay_p
                else:
                    lam = ad.sigmoid(p["b_lambda"])
                    P_decayed = ad.scale(P, lam)
                    lam_f = float(lam.data[0])

                # --- write into Persistent Memory ---
                if self.pipeline_mode == "direct":
                    # Phase 4.1/4.2 style: instantaneous outer(k,v) gated by promote
                    write = ad.scale(ad.outer(k, v), p_promote)
                    P = ad.add(P_decayed, write)
                else:
                    # True T→P: promote a transform of the accumulated trace
                    # content into P, then soft-clear the promoted mass from T
                    # so information *moves* rather than being re-copied every
                    # step (which flooded P and collapsed learning).
                    tk = ad.matvec(p["W_tk"], T_k)
                    tv = ad.matvec(p["W_tv"], T_v)
                    # scale outer by 1/sqrt(d) to match attention-style magnitudes
                    write = ad.mul_const(ad.scale(ad.outer(tk, tv), p_promote),
                                         1.0 / (self.d_h ** 0.5))
                    P = ad.add(P_decayed, write)
                    # soft-clear: T ← (1 - p_promote) * T
                    # retain unpromoted residual evidence in the trace
                    one = ad.Tensor(np.array([1.0]))
                    retain = ad.add(one, ad.mul_const(p_promote, -1.0))  # 1 - p_promote
                    T_k = ad.scale(T_k, retain)
                    T_v = ad.scale(T_v, retain)
                    T_sum = ad.scale(T_sum, retain)

                # logging
                g_keep_f = float(g_keep.data[0])
                p_prom_f = float(p_promote.data[0])
                self.last_keep_values.append(g_keep_f)
                self.last_gate_values.append(p_prom_f)
                self.last_lambda_values.append(lam_f)

                is_key = t in oracle_write
                if is_key:
                    self.diag["promote_key"].append(p_prom_f)
                    self.diag["keep_key"].append(g_keep_f)
                else:
                    self.diag["promote_filler"].append(p_prom_f)
                    self.diag["keep_filler"].append(g_keep_f)
                self.diag["lambda_all"].append(lam_f)
            else:
                self.last_keep_values.append(0.0)
                self.last_gate_values.append(0.0)
                self.last_lambda_values.append(self.decay_p)

            if t in qpos:
                q = ad.matvec(p["Wq"], emb)
                read = ad.vecmat(q, P)
                comb = ad.concat(S, read)
                hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
                logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

            prev_emb = emb

        self.diag["trace_norm"].append(float(np.linalg.norm(T_sum.data)))
        self.diag["mem_norm"].append(float(np.linalg.norm(P.data)))
        if self.pipeline_mode == "trace":
            self.diag["tk_norm"].append(float(np.linalg.norm(T_k.data)))
            self.diag["tv_norm"].append(float(np.linalg.norm(T_v.data)))
        return self._finish(losses, correct, total, train)

    def _finish(self, losses, correct, total, train):
        if not losses:
            return 0.0, 0.0
        total_loss = losses[0]
        for l in losses[1:]:
            total_loss = ad.add(total_loss, l)
        if train:
            total_loss.backward()
        else:
            ad.free_graph(total_loss)
        return float(total_loss.data) / total, correct / total

    def reset_diag(self):
        for k in self.diag:
            self.diag[k] = []

    def summarize_diag(self):
        def _m(xs):
            return float(np.mean(xs)) if xs else None
        out = {
            "promote_key": _m(self.diag["promote_key"]),
            "promote_filler": _m(self.diag["promote_filler"]),
            "keep_key": _m(self.diag["keep_key"]),
            "keep_filler": _m(self.diag["keep_filler"]),
            "lambda_mean": _m(self.diag["lambda_all"]),
            "trace_norm": _m(self.diag["trace_norm"]),
            "mem_norm": _m(self.diag["mem_norm"]),
            "tk_norm": _m(self.diag["tk_norm"]),
            "tv_norm": _m(self.diag["tv_norm"]),
        }
        if self.diag["promote_key"] and self.diag["promote_filler"]:
            out["promote_ratio"] = (
                out["promote_key"] / max(out["promote_filler"], 1e-8)
            )
        return out


class AACDiffLifecycle:
    """Phase 4.4: trainable memory lifecycle on top of utility + global λ.

    Lifecycle (conceptual):
      WRITE → CONFIDENCE → REINFORCE → DECAY/RETAIN → EVICT

    Modes (`lifecycle_mode`):
      "none"            — write + global λ only (Phase 4.2/4.3 default)
      "reinforce"       — + learned reinforce on associative reads
      "reinforce_evict" — + learned reinforce and a learned retention gate
                          (soft eviction / extra decay beyond λ)

    Write path: direct outer(k,v) * p_promote (winner of Phase 4.3).
    T remains the evidence buffer for Value (not the content source for P).
    No oracle training targets.
    """
    name = "AAC-diff lifecycle (write + reinforce + retain/evict)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, decay_p=0.999,
                 decay_t=0.9, value_hidden=16, lifecycle_mode="none",
                 seed=0):
        if lifecycle_mode not in ("none", "reinforce", "reinforce_evict"):
            raise ValueError(
                f"lifecycle_mode must be none|reinforce|reinforce_evict, got {lifecycle_mode!r}")
        rng = np.random.default_rng(seed)
        self.params = {
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Whh": _init(rng, (d_h, d_h)),
            "Wxh": _init(rng, (d_h, d_emb)),
            "bh": _zeros((d_h,)),
            "Wk": _init(rng, (d_h, d_emb)),
            "Wv": _init(rng, (d_h, d_emb)),
            "Wq": _init(rng, (d_h, d_emb)),
            "W_keep": _init(rng, (1, d_h + d_emb)),
            "b_keep": ad.Tensor(np.array([0.0]), requires_grad=True),
            "Wv1": _init(rng, (value_hidden, 2 * d_h + d_emb)),
            "bv1": _zeros((value_hidden,)),
            "Wv2": _init(rng, (1, value_hidden)),
            "bv2": ad.Tensor(np.array([0.0]), requires_grad=True),
            "Wcomb": _init(rng, (d_h, 2 * d_h)),
            "bcomb": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        # global persistence λ
        p0 = float(np.clip(decay_p, 1e-4, 1 - 1e-4))
        b0 = np.log(p0 / (1 - p0))
        self.params["b_lambda"] = ad.Tensor(np.array([b0]), requires_grad=True)

        if lifecycle_mode in ("reinforce", "reinforce_evict"):
            # reinforce strength: how much to boost (q, read) association after retrieval
            self.params["W_reinf"] = _init(rng, (1, d_h + d_emb))
            self.params["b_reinf"] = ad.Tensor(np.array([-1.0]), requires_grad=True)  # start soft
        if lifecycle_mode == "reinforce_evict":
            # retention gate beyond λ: low → soft-evict directions (global scalar first)
            self.params["W_retain"] = _init(rng, (1, d_h + d_emb))
            self.params["b_retain"] = ad.Tensor(np.array([2.0]), requires_grad=True)  # start near 1

        self.opt = AdamTensor(self.params)
        self.d_h = d_h
        self.d_emb = d_emb
        self.decay_p = decay_p
        self.decay_t = decay_t
        self.lifecycle_mode = lifecycle_mode

        self.last_gate_values = []
        self.last_keep_values = []
        self.last_lambda_values = []
        self.last_reinf_values = []
        self.last_retain_values = []
        self.diag = {
            "promote_key": [], "promote_filler": [],
            "keep_key": [], "keep_filler": [],
            "lambda_all": [], "reinf_all": [], "retain_all": [],
            "trace_norm": [], "mem_norm": [],
            "reinforce_count": 0,
        }

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        oracle_write = set(
            int(pos) for pos in
            [kp + 1 for kp in episode["key_positions"] if kp + 1 < len(tokens)]
        )

        S = ad.Tensor(np.zeros(self.d_h))
        T = ad.Tensor(np.zeros(self.d_h))
        P = ad.Tensor(np.zeros((self.d_h, self.d_h)))

        losses, correct, total = [], 0, 0
        self.last_gate_values = []
        self.last_keep_values = []
        self.last_lambda_values = []
        self.last_reinf_values = []
        self.last_retain_values = []
        reinf_count = 0
        prev_emb = None

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            S = ad.tanh(ad.add(ad.add(ad.matvec(p["Whh"], S),
                                       ad.matvec(p["Wxh"], emb)), p["bh"]))

            if prev_emb is not None:
                k = ad.matvec(p["Wk"], prev_emb)
                v = ad.matvec(p["Wv"], emb)

                # keep → T (evidence for Value)
                keep_in = ad.concat(S, emb)
                g_keep = ad.sigmoid(ad.add(ad.matvec(p["W_keep"], keep_in), p["b_keep"]))
                T = ad.add(ad.mul_const(T, self.decay_t), ad.scale(v, g_keep))

                # Value → promote
                val_in = ad.concat(ad.concat(S, T), emb)
                h_v = ad.tanh(ad.add(ad.matvec(p["Wv1"], val_in), p["bv1"]))
                v_score = ad.add(ad.matvec(p["Wv2"], h_v), p["bv2"])
                p_promote = ad.sigmoid(v_score)

                # λ decay
                lam = ad.sigmoid(p["b_lambda"])
                P_decayed = ad.scale(P, lam)
                lam_f = float(lam.data[0])

                # optional learned retention (soft eviction factor)
                retain_f = 1.0
                if self.lifecycle_mode == "reinforce_evict":
                    rin = ad.concat(S, emb)
                    g_retain = ad.sigmoid(
                        ad.add(ad.matvec(p["W_retain"], rin), p["b_retain"]))
                    P_decayed = ad.scale(P_decayed, g_retain)
                    retain_f = float(g_retain.data[0])
                    self.last_retain_values.append(retain_f)
                    self.diag["retain_all"].append(retain_f)

                # write
                write = ad.scale(ad.outer(k, v), p_promote)
                P = ad.add(P_decayed, write)

                g_keep_f = float(g_keep.data[0])
                p_prom_f = float(p_promote.data[0])
                self.last_keep_values.append(g_keep_f)
                self.last_gate_values.append(p_prom_f)
                self.last_lambda_values.append(lam_f)

                is_key = t in oracle_write
                if is_key:
                    self.diag["promote_key"].append(p_prom_f)
                    self.diag["keep_key"].append(g_keep_f)
                else:
                    self.diag["promote_filler"].append(p_prom_f)
                    self.diag["keep_filler"].append(g_keep_f)
                self.diag["lambda_all"].append(lam_f)
            else:
                self.last_keep_values.append(0.0)
                self.last_gate_values.append(0.0)
                self.last_lambda_values.append(self.decay_p)

            # --- read + optional reinforce at query positions ---
            if t in qpos:
                q = ad.matvec(p["Wq"], emb)
                read = ad.vecmat(q, P)

                if self.lifecycle_mode in ("reinforce", "reinforce_evict"):
                    # learned reinforce strength from (S, emb)
                    rin = ad.concat(S, emb)
                    g_reinf = ad.sigmoid(
                        ad.add(ad.matvec(p["W_reinf"], rin), p["b_reinf"]))
                    # Hebbian strengthen of the retrieved association
                    # P ← P + g_reinf * outer(q, read) / sqrt(d)
                    boost = ad.mul_const(
                        ad.scale(ad.outer(q, read), g_reinf),
                        1.0 / (self.d_h ** 0.5))
                    P = ad.add(P, boost)
                    reinf_f = float(g_reinf.data[0])
                    self.last_reinf_values.append(reinf_f)
                    self.diag["reinf_all"].append(reinf_f)
                    reinf_count += 1

                comb = ad.concat(S, read)
                hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
                logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

            prev_emb = emb

        self.diag["trace_norm"].append(float(np.linalg.norm(T.data)))
        self.diag["mem_norm"].append(float(np.linalg.norm(P.data)))
        self.diag["reinforce_count"] = reinf_count
        return self._finish(losses, correct, total, train)

    def _finish(self, losses, correct, total, train):
        if not losses:
            return 0.0, 0.0
        total_loss = losses[0]
        for l in losses[1:]:
            total_loss = ad.add(total_loss, l)
        if train:
            total_loss.backward()
        else:
            ad.free_graph(total_loss)
        return float(total_loss.data) / total, correct / total

    def reset_diag(self):
        for k in self.diag:
            if k == "reinforce_count":
                self.diag[k] = 0
            else:
                self.diag[k] = []

    def summarize_diag(self):
        def _m(xs):
            return float(np.mean(xs)) if xs else None
        out = {
            "promote_key": _m(self.diag["promote_key"]),
            "promote_filler": _m(self.diag["promote_filler"]),
            "keep_key": _m(self.diag["keep_key"]),
            "keep_filler": _m(self.diag["keep_filler"]),
            "lambda_mean": _m(self.diag["lambda_all"]),
            "reinf_mean": _m(self.diag["reinf_all"]),
            "retain_mean": _m(self.diag["retain_all"]),
            "trace_norm": _m(self.diag["trace_norm"]),
            "mem_norm": _m(self.diag["mem_norm"]),
            "reinforce_count": self.diag["reinforce_count"],
        }
        if self.diag["promote_key"] and self.diag["promote_filler"]:
            out["promote_ratio"] = (
                out["promote_key"] / max(out["promote_filler"], 1e-8)
            )
        return out


class AACDiffFinal(AACDiffLifecycle):
    """Phase 4.5: canonical AAC stack from Phases 4.1–4.4 winners.

    - Learned utility promotion (Value_θ → p_promote)  [4.1]
    - Learned global persistence λ                     [4.2]
    - Direct write of current binding into P           [4.3]
    - No matrix-level reinforce / soft-evict           [4.4]
    - T as evidence buffer for Value (T ≠ P)

    Equivalent to AACDiffLifecycle(lifecycle_mode="none").
    """
    name = "AAC-final (utility + global λ + direct write)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, decay_p=0.999,
                 decay_t=0.9, value_hidden=16, seed=0):
        super().__init__(
            vocab_size=vocab_size, d_emb=d_emb, d_h=d_h,
            decay_p=decay_p, decay_t=decay_t, value_hidden=value_hidden,
            lifecycle_mode="none", seed=seed,
        )
