# Extracted from build_evidence.py for portable use. See manifests/code_origins.csv.
import numpy as np

# Historical key name: these fixed values are kernel sharpness beta, not fitted temperatures.
cfg = {"temperatures": [1, 5, 10]}


def summarize(sim, allow, labels):
    """Summarize one endpoint side before any fit-only standardization.

    sim: (n_queries, n_events) cosines; allow: matching admissibility mask.
    labels: (n_events,) integers in query-direction order [=, >, <, ~].
    Return counts (n_queries, 4), Q and S (n_queries, 3, 4),
    log(1+ESS) and entropy (n_queries, 3), for beta=1,5,10.
    Q divides type mass by all-type mass; S divides by that type's count.
    An empty admissible neighbourhood returns zero raw summaries.
    """
    one = np.eye(4)[labels]
    counts = allow @ one
    qs = []
    strength = []
    ess = []
    ent = []
    for beta in cfg["temperatures"]:
        w = np.exp(beta * (sim - 1)) * allow
        total = w.sum(1)
        mass = w @ one
        q = mass / total.clip(1e-30)[:, None]
        qs.append(q)
        strength.append(mass / counts.clip(1))
        ess.append(np.log1p(total**2 / (w * w).sum(1).clip(1e-30)))
        ent.append(-(q * np.log(q.clip(1e-30))).sum(1))
    return (
        counts,
        np.stack(qs, 1),
        np.stack(strength, 1),
        np.stack(ess, 1),
        np.stack(ent, 1),
    )
