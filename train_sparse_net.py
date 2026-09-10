"""
Tiny non-negative, sparsely-activating network.

Task: predict the next symbol in a sequence drawn from an order-1 Markov
chain whose "peakiness" p controls predictability.
  p = 1/8 (uniform)  -> fully unpredictable
  p -> 1.0           -> nearly deterministic, highly predictable

Network: Linear -> ReLU (non-negative) -> Linear -> softmax.
We add an L1 penalty on the (non-negative) hidden activations during
training. Critically, we do NOT use top-k / a fixed quota: how many
hidden units end up firing (nonzero) is left free to emerge from the
input, so we can test whether it tracks predictability on its own.
"""
import numpy as np
import json

rng = np.random.default_rng(0)

ALPHABET = 8
WINDOW = 3
INPUT_DIM = ALPHABET * WINDOW
HIDDEN = 200
LR = 0.05
L1_LAMBDA = 0.004
EPOCHS = 4000
BATCH = 256
SHIFT = 1  # "preferred next symbol" rule: next = (current + SHIFT) mod ALPHABET


def gen_batch(coherence, batch=BATCH, window=WINDOW):
    """
    Sample `batch` (context, next_symbol) pairs.
    `coherence` controls, at each step, the probability that the sequence
    follows the cyclic rule next=(prev+SHIFT)%ALPHABET rather than jumping
    randomly. Critically, whether a given WINDOW actually looks cyclic
    (0, 1, or 2 of its own internal steps match the rule) is something the
    network can see directly in the one-hot input -- it is not hidden
    information about the generating process. That's what lets hidden-unit
    activity legitimately track predictability instance-by-instance.
    """
    seqs = np.zeros((batch, window), dtype=int)
    seqs[:, 0] = rng.integers(0, ALPHABET, size=batch)
    for i in range(1, window):
        follows_rule = rng.random(batch) < coherence
        cyclic_next = (seqs[:, i - 1] + SHIFT) % ALPHABET
        rand_next = rng.integers(0, ALPHABET, size=batch)
        seqs[:, i] = np.where(follows_rule, cyclic_next, rand_next)

    target_follows = rng.random(batch) < coherence
    cyclic_target = (seqs[:, -1] + SHIFT) % ALPHABET
    rand_target = rng.integers(0, ALPHABET, size=batch)
    nxt = np.where(target_follows, cyclic_target, rand_target)

    x = np.zeros((batch, window, ALPHABET))
    x[np.arange(batch)[:, None], np.arange(window)[None, :], seqs] = 1.0
    x = x.reshape(batch, -1)
    y = np.zeros((batch, ALPHABET))
    y[np.arange(batch), nxt] = 1.0
    return x, y, seqs, nxt


def match_count(seqs):
    """How many consecutive steps in each window already follow the cyclic
    rule -- the visible, instance-level predictability signal."""
    matches = np.zeros(len(seqs), dtype=int)
    for i in range(1, seqs.shape[1]):
        matches += (seqs[:, i] == (seqs[:, i - 1] + SHIFT) % ALPHABET).astype(int)
    return matches


def init_params():
    W1 = rng.normal(0, 1 / np.sqrt(INPUT_DIM), (INPUT_DIM, HIDDEN))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, 1 / np.sqrt(HIDDEN), (HIDDEN, ALPHABET))
    b2 = np.zeros(ALPHABET)
    return W1, b1, W2, b2


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    h = np.maximum(z1, 0)
    z2 = h @ W2 + b2
    y_hat = softmax(z2)
    return z1, h, z2, y_hat


def train():
    W1, b1, W2, b2 = init_params()
    for epoch in range(EPOCHS):
        coherence = rng.uniform(0.0, 0.97)  # sample a predictability level
        x, y, _, _ = gen_batch(coherence)
        N = x.shape[0]

        z1, h, z2, y_hat = forward(x, W1, b1, W2, b2)

        dz2 = (y_hat - y) / N
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0)

        dh = dz2 @ W2.T + (L1_LAMBDA / N)
        dz1 = dh * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        W1 -= LR * dW1; b1 -= LR * db1
        W2 -= LR * dW2; b2 -= LR * db2

        if epoch % 500 == 0:
            loss = -np.sum(y * np.log(y_hat + 1e-9)) / N
            active = (h > 1e-6).mean()
            print(f"epoch {epoch:5d}  coherence={coherence:.2f}  loss={loss:.3f}  mean_active_frac={active:.3f}")
    return W1, b1, W2, b2


def evaluate(W1, b1, W2, b2, n_levels=11, trials=600):
    """For each nominal coherence level, measure mean active-unit fraction
    (aggregate view) AND bucket by the window's own visible match count
    0/1/2 (the instance-level view -- this is the one that should show the
    cleanest relationship, since it's the feature actually visible to the
    network)."""
    curve = []
    for coherence in np.linspace(0.0, 0.97, n_levels):
        x, y, seqs, _ = gen_batch(coherence, batch=trials)
        _, h, _, y_hat = forward(x, W1, b1, W2, b2)
        active_frac = (h > 1e-6).mean(axis=1).mean()
        acc = (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()
        curve.append({"coherence": round(float(coherence), 3),
                      "active_frac": round(float(active_frac), 4),
                      "accuracy": round(float(acc), 4)})

    by_match = []
    x, y, seqs, _ = gen_batch(0.5, batch=4000)  # mixed pool, then bucket by observed matches
    _, h, _, y_hat = forward(x, W1, b1, W2, b2)
    m = match_count(seqs)
    for k in range(WINDOW):  # 0 .. WINDOW-1 matches
        mask = m == k
        if mask.sum() < 5:
            continue
        by_match.append({"matches": int(k),
                          "n": int(mask.sum()),
                          "active_frac": round(float((h[mask] > 1e-6).mean()), 4),
                          "confidence": round(float(y_hat[mask].max(axis=1).mean()), 4)})
    return curve, by_match


if __name__ == "__main__":
    W1, b1, W2, b2 = train()
    curve, by_match = evaluate(W1, b1, W2, b2)
    print(json.dumps({"curve": curve, "by_match": by_match}, indent=2))

    out = {
        "meta": {"alphabet": ALPHABET, "window": WINDOW, "hidden": HIDDEN, "shift": SHIFT},
        "W1": W1.tolist(), "b1": b1.tolist(),
        "W2": W2.tolist(), "b2": b2.tolist(),
        "eval_curve": curve,
        "by_match": by_match,
    }
    with open("sparse_net_weights.json", "w") as f:
        json.dump(out, f)
    print("Saved weights + eval curve to sparse_net_weights.json")
