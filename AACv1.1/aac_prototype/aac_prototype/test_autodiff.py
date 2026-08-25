"""
Numerical gradient checking for aac/autodiff.py. This must pass before
any model built on top of the autodiff engine can be trusted -- hand-
derived backward passes are exactly the kind of code that looks right and
silently isn't.

Method: central finite differences on each input element, compared
against the analytic gradient from .backward().
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from aac import autodiff as ad


def numerical_grad(f, x, eps=1e-5):
    """f: python callable, numpy array -> scalar. Returns numeric grad,
    same shape as x, via central differences."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def check(name, build_graph, inputs, tol=1e-4):
    """build_graph(tensors) -> scalar Tensor (the loss to backprop from).
    inputs: dict name -> numpy array (will be wrapped as leaf Tensors)."""
    tensors = {k: ad.Tensor(v.copy(), requires_grad=True) for k, v in inputs.items()}
    out = build_graph(tensors)
    for t in tensors.values():
        t.zero_grad()
    out.backward()
    analytic = {k: tensors[k].grad.copy() for k in tensors}

    ok = True
    for k, arr in inputs.items():
        def f(x, k=k):
            local = {kk: ad.Tensor(vv.copy()) for kk, vv in inputs.items()}
            local[k] = ad.Tensor(x)
            return float(build_graph(local).data)
        numeric = numerical_grad(f, arr.copy())
        diff = np.max(np.abs(numeric - analytic[k]))
        denom = max(1e-6, np.max(np.abs(numeric)))
        rel = diff / denom
        status = "OK" if rel < tol else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {name}:{k}  max_abs_diff={diff:.2e}  rel={rel:.2e}")
    return ok


def main():
    rng = np.random.default_rng(0)
    all_ok = True

    all_ok &= check(
        "matvec+tanh",
        lambda t: ad.tanh(ad.matvec(t["W"], t["x"])).data.sum() and
        _sumT(ad.tanh(ad.matvec(t["W"], t["x"]))),
        {"W": rng.normal(size=(4, 3)), "x": rng.normal(size=3)},
    )

    all_ok &= check(
        "vecmat+sigmoid",
        lambda t: _sumT(ad.sigmoid(ad.vecmat(t["x"], t["W"]))),
        {"x": rng.normal(size=3), "W": rng.normal(size=(3, 5))},
    )

    all_ok &= check(
        "outer+relu",
        lambda t: _sumT(ad.relu(ad.outer(t["a"], t["b"]))),
        {"a": rng.normal(size=3), "b": rng.normal(size=4)},
    )

    all_ok &= check(
        "dot",
        lambda t: ad.dot(t["a"], t["b"]),
        {"a": rng.normal(size=5), "b": rng.normal(size=5)},
    )

    all_ok &= check(
        "sub",
        lambda t: _sumT(ad.sub(t["a"], t["b"])),
        {"a": rng.normal(size=4), "b": rng.normal(size=4)},
    )

    all_ok &= check(
        "softmax_cross_entropy",
        lambda t: ad.softmax_cross_entropy(t["logits"], 2),
        {"logits": rng.normal(size=6)},
    )

    all_ok &= check(
        "concat+scale",
        lambda t: _sumT(ad.scale(ad.concat(t["a"], t["b"]), _const(0.37))),
        {"a": rng.normal(size=3), "b": rng.normal(size=4)},
    )

    all_ok &= check(
        "stack+vecmat (attention-like read)",
        lambda t: _sumT(ad.vecmat(
            ad.softmax(ad.matvec(ad.stack([t["k0"], t["k1"], t["k2"]]), t["q"])),
            ad.stack([t["v0"], t["v1"], t["v2"]]))),
        {"q": rng.normal(size=4), "k0": rng.normal(size=4), "k1": rng.normal(size=4),
         "k2": rng.normal(size=4), "v0": rng.normal(size=3), "v1": rng.normal(size=3),
         "v2": rng.normal(size=3)},
    )

    # ---- the case that actually matters: multi-step BPTT through a
    # recurrence + a fast-weight memory, exactly the pattern used in the
    # real models. If this passes, the models are trustworthy.
    all_ok &= check(
        "3-step RNN + gated fast-weight memory (full BPTT)",
        _three_step_recurrence,
        {
            "Whh": rng.normal(scale=0.3, size=(4, 4)),
            "Wxh": rng.normal(scale=0.3, size=(4, 3)),
            "Wk": rng.normal(scale=0.3, size=(4, 4)),
            "Wv": rng.normal(scale=0.3, size=(4, 4)),
            "Wg": rng.normal(scale=0.3, size=(1, 4)),
            "x0": rng.normal(size=3), "x1": rng.normal(size=3), "x2": rng.normal(size=3),
        },
    )

    # ---- delta-rule update with subtraction (AAC-diff with delta-rule)
    all_ok &= check(
        "3-step RNN + delta-rule fast-weight memory (with sub)",
        _three_step_recurrence_delta,
        {
            "Whh": rng.normal(scale=0.3, size=(4, 4)),
            "Wxh": rng.normal(scale=0.3, size=(4, 3)),
            "Wk": rng.normal(scale=0.3, size=(4, 4)),
            "Wv": rng.normal(scale=0.3, size=(4, 4)),
            "Wg": rng.normal(scale=0.3, size=(1, 4)),
            "x0": rng.normal(size=3), "x1": rng.normal(size=3), "x2": rng.normal(size=3),
        },
    )

    print("\nALL PASS" if all_ok else "\nSOME CHECKS FAILED")
    return all_ok


def _sumT(t):
    out = ad.Tensor(t.data.sum(), _children=(t,), _op="sum")

    def _backward():
        if t.requires_grad:
            t.grad += np.ones_like(t.data) * out.grad
    out._backward = _backward
    return out


def _const(c):
    return ad.Tensor(np.array(c))


def _three_step_recurrence(t):
    d = 4
    h = ad.Tensor(np.zeros(d))
    M = ad.Tensor(np.zeros((d, d)))
    loss_terms = []
    for x in (t["x0"], t["x1"], t["x2"]):
        h = ad.tanh(ad.add(ad.matvec(t["Whh"], h), ad.matvec(t["Wxh"], x)))
        k = ad.matvec(t["Wk"], h)
        v = ad.matvec(t["Wv"], h)
        g = ad.sigmoid(ad.matvec(t["Wg"], h))  # shape (1,)
        M = ad.add(ad.mul_const(M, 0.9), ad.scale(ad.outer(k, v), g))
        read = ad.vecmat(h, M)
        loss_terms.append(_sumT(read))
    total = loss_terms[0]
    for lt in loss_terms[1:]:
        total = ad.add(total, lt)
    return total


def _three_step_recurrence_delta(t):
    """Same as _three_step_recurrence but using delta-rule memory update:
    M_t = decay*M_{t-1} + g*outer(k,v) - g*beta*outer(k, k@M_{t-1})"""
    d = 4
    h = ad.Tensor(np.zeros(d))
    M = ad.Tensor(np.zeros((d, d)))
    loss_terms = []
    decay = 0.9
    beta = 0.1
    for x in (t["x0"], t["x1"], t["x2"]):
        h = ad.tanh(ad.add(ad.matvec(t["Whh"], h), ad.matvec(t["Wxh"], x)))
        k = ad.matvec(t["Wk"], h)
        v = ad.matvec(t["Wv"], h)
        g = ad.sigmoid(ad.matvec(t["Wg"], h))  # shape (1,)
        # Delta-rule: subtract out old association before adding new one
        kM = ad.vecmat(k, M)  # k @ M
        delta_correction = ad.scale(ad.outer(k, kM), ad.mul_const(g, beta))
        M = ad.sub(ad.add(ad.mul_const(M, decay), ad.scale(ad.outer(k, v), g)), 
                   delta_correction)
        read = ad.vecmat(h, M)
        loss_terms.append(_sumT(read))
    total = loss_terms[0]
    for lt in loss_terms[1:]:
        total = ad.add(total, lt)
    return total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
