import json

# Extracted from legacy_lrpca.py for portable use. See manifests/code_origins.csv.
import torch
from torch import nn
import numpy as np
import math
import time


def retract(l):
    q, r = torch.linalg.qr(l, mode="reduced")
    s = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
    return q * s


def solve(X, XT, mu, norm2, y, L, log=None, outer=4, steps=10):
    """Fit the retained classification/reconstruction low-rank objective.

    X/XT are fit-only sparse text tensors and their transpose, mu the fit
    mean, norm2 the centered fit squared norm, y the four-class fit labels,
    and L the initial rank-64 basis. Return basis, temporary class head,
    class intercept and optimization records. Neither evaluation labels
    nor evaluation-driven stopping enter this computation.
    """
    n = len(y)
    Y = torch.nn.functional.one_hot(y, 4).to(torch.float64)
    w = torch.zeros((64, 4), device=L.device, dtype=L.dtype)
    c = torch.log(Y.mean(0))
    records = []
    headcalls = []
    closurecalls = 0
    updates = 0

    def proj(l):
        return torch.sparse.mm(X, l) - mu @ l

    def aux(z):
        return z @ w + c

    def objective(z):
        ce = torch.nn.functional.cross_entropy(aux(z), y)
        rec = 1 - torch.square(z).sum() / norm2
        return 0.5 * ce + 0.5 * rec, ce, rec

    def fithead(z, oi):
        nonlocal w, c, closurecalls
        wp = nn.Parameter(w.detach().clone())
        cp = nn.Parameter(c.detach().clone())
        pars = [wp, cp]
        opt = torch.optim.LBFGS(
            pars,
            lr=1.0,
            max_iter=100,
            max_eval=125,
            tolerance_grad=1e-7,
            tolerance_change=1e-10,
            history_size=20,
            line_search_fn="strong_wolfe",
        )
        calls = 0
        t = time.perf_counter()

        def closure():
            nonlocal calls, closurecalls
            opt.zero_grad(set_to_none=True)
            pred = z @ wp + cp
            loss = torch.nn.functional.cross_entropy(pred, y)
            assert torch.isfinite(loss)
            loss.backward()
            calls += 1
            closurecalls += 1
            return loss

        opt.step(closure)
        w = wp.detach()
        c = cp.detach()
        ce = float(torch.nn.functional.cross_entropy(aux(z), y))
        grad = max(float(p.grad.abs().max()) for p in pars)
        headcalls.append(
            dict(
                outer=oi,
                optimizer="LBFGS",
                max_iterations=100,
                closure_evaluations=calls,
                CE=ce,
                last_closure_gradient_max=grad,
                seconds=time.perf_counter() - t,
                head_parameters=sum(p.numel() for p in pars),
            )
        )

    z = proj(L)
    fithead(z, 0)
    alpha = 1.0
    for oi in range(1, outer + 1):
        for si in range(1, steps + 1):
            t = time.perf_counter()
            val, ce, rec = objective(z)
            P = torch.softmax(aux(z), 1)
            res = (P - Y) / n
            g = (
                0.5
                * (torch.sparse.mm(XT, res @ w.T) - mu[:, None] * (res @ w.T).sum(0))
                - (torch.sparse.mm(XT, z) - mu[:, None] * z.sum(0)) / norm2
            )
            g = g - L @ (L.T @ g)
            gn = float(torch.linalg.norm(g))
            val0 = float(val)
            accepted = False
            trialalpha = min(alpha * 2, 64.0)
            ls = []
            if gn > 1e-8:
                for bt in range(16):
                    Ln = retract(L - trialalpha * g)
                    zn = proj(Ln)
                    vn, _, _ = objective(zn)
                    vnew = float(vn)
                    ls.append([trialalpha, vnew])
                    if (
                        math.isfinite(vnew)
                        and vnew <= val0 - 1e-4 * trialalpha * gn * gn
                    ):
                        accepted = True
                        break
                    trialalpha *= 0.5
            if accepted:
                L = Ln
                z = zn
                alpha = trialalpha
                updates += 1
            row = dict(
                outer=oi,
                step=si,
                objective_before=val0,
                CE_before=float(ce),
                reconstruction_fraction_before=float(rec),
                gradient_norm=gn,
                accepted=accepted,
                step_size=trialalpha,
                backtracking=ls,
                objective_after=float(objective(z)[0]),
                orthogonality_error=float(
                    (L.T @ L - torch.eye(64, device=L.device, dtype=L.dtype))
                    .abs()
                    .max()
                ),
                seconds=time.perf_counter() - t,
            )
            records.append(row)
            if log:
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
            if not accepted:
                break
        fithead(z, oi)
        print(
            json.dumps(
                dict(
                    outer=oi,
                    objective=float(objective(z)[0]),
                    CE=float(objective(z)[1]),
                    retained=1 - float(objective(z)[2]),
                    updates=updates,
                )
            ),
            flush=True,
        )
    return (
        L,
        w,
        c,
        dict(
            steps=records,
            head_calls=headcalls,
            head_refit_calls=len(headcalls),
            closure_evaluations=closurecalls,
            accepted_projection_updates=updates,
            final_objective=float(objective(z)[0]),
            final_CE=float(objective(z)[1]),
            final_reconstruction_fraction=float(objective(z)[2]),
        ),
    )
