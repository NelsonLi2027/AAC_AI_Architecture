"""
Three models, all trained end-to-end via full backprop-through-time using
aac/autodiff.py, all sharing the same embedding dim (d_emb) and hidden
dim (d_h), all trained on identical episodes with identical Adam settings
-- a matched-budget comparison, not a cherry-picked one.

  RNNBaseline      -- a plain tanh-RNN + linear readout. No external
                      memory of any kind. This is the "what if you don't
                      bother with any of AAC's machinery" floor.

  AttentionBaseline -- a single-layer causal self-attention block (the
                      thing AAC's whole premise is to avoid the cost of).
                      IMPORTANT ASYMMETRY, stated plainly: this model's
                      "memory" is the full, uncompressed key/value history
                      -- it never forgets or compresses anything, so its
                      effective capacity grows with sequence length. It is
                      a ceiling on achievable accuracy, not a fair
                      apples-to-apples capacity match.

  AACDiffModel     -- fast recurrent state h_t (AAC's S_t) + a gated,
                      FIXED-SIZE associative fast-weight matrix M (AAC's
                      P_t, relaxed from discrete write/promote/evict into
                      a continuous gated update so it's differentiable).
                      Capacity is bounded by d_h x d_h regardless of
                      sequence length -- unlike the attention baseline,
                      this model has to actually compress.

All three read the write/read gate scalars through a sigmoid, so nothing
here is a hard discrete decision -- that is the explicit scope trade-off
made to get real end-to-end gradients through the memory mechanism (see
README's "what changed to make this run" section).
"""
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


def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


# ----------------------------------------------------------------------
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
            total_loss.backward()  # accumulates into param.grad; caller
            # decides when to opt.step()/opt.zero_grad() (see
            # train_compare.py's mini-batch loop) -- stepping after every
            # single noisy episode was the main cause of slow learning.
        return float(total_loss.data) / total, correct / total


# ----------------------------------------------------------------------
class AttentionBaseline:
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
                # it, rather than the key's own embedding again.
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


# ----------------------------------------------------------------------
class AACDiffModel:
    name = "AAC-diff (fast state + gated fixed-size associative memory)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, decay=0.95, beta=0.1,
                 update_rule="hebbian", gate_mode="learned", seed=0):
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
            "Wcomb": _init(rng, (d_h, 2 * d_h)),
            "bcomb": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        self.opt = AdamTensor(self.params)
        self.d_h = d_h
        self.decay = decay
        self.beta = beta  # for regularized delta-rule
        self.update_rule = update_rule  # "hebbian", "delta_regularized", or "widrow_hoff"
        self.gate_mode = gate_mode      # "learned" or "oracle"
        self.last_gate_values = []      # for reporting how often it "writes"
        self.grad_norms = {"Wg": [], "Wk": [], "Wv": [], "Wq": []}  # for diagnostics

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        oracle_positions = set(int(pos) for pos in [kp + 1 for kp in episode["key_positions"] if kp + 1 < len(tokens)])
        h = ad.Tensor(np.zeros(self.d_h))
        M = ad.Tensor(np.zeros((self.d_h, self.d_h)))
        losses, correct, total = [], 0, 0
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
                    g = ad.Tensor(np.array([1.0 if t in oracle_positions else 0.0], dtype=np.float64))
                else:
                    g = ad.sigmoid(ad.matvec(p["Wg"], h))
                self.last_gate_values.append(float(g.data[0]))

                # Memory update rule selection
                if self.update_rule == "hebbian":
                    if self.gate_mode == "oracle":
                        if t in oracle_positions:
                            M = ad.add(M, ad.outer(k, v))
                    else:
                        M = ad.add(ad.mul_const(M, self.decay), ad.scale(ad.outer(k, v), g))

                elif self.update_rule == "delta_regularized":
                    if self.gate_mode == "oracle":
                        if t in oracle_positions:
                            M = ad.add(M, ad.outer(k, v))
                    else:
                        kM = ad.vecmat(k, M)
                        delta_correction = ad.scale(ad.outer(k, kM), ad.mul_const(g, self.beta))
                        M = ad.sub(ad.add(ad.mul_const(M, self.decay), ad.scale(ad.outer(k, v), g)),
                                   delta_correction)

                elif self.update_rule == "widrow_hoff":
                    if self.gate_mode == "oracle":
                        if t in oracle_positions:
                            M = ad.add(M, ad.outer(k, v))
                    else:
                        kM = ad.vecmat(k, M)
                        error = ad.sub(v, kM)
                        M = ad.add(M, ad.scale(ad.outer(k, error), g))
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

        result = RNNBaseline._finish(self, losses, correct, total, train)
        if train and self.opt.t <= 500 and self.opt.t % 50 == 0:
            for key in ["Wg", "Wk", "Wv", "Wq"]:
                if key in p:
                    gnorm = float(np.linalg.norm(p[key].grad))
                    self.grad_norms[key].append((self.opt.t, gnorm))
        return result
