"""Métriques d'évaluation.

`gap` reprend la définition du Met Dataset (Ypsilantis et al., NeurIPS 2021, code MIT :
https://github.com/nikosips/met) : les distracteurs ont le label None.
"""
from __future__ import annotations

import numpy as np


def gap(pred: list, conf: list[float], truth: list) -> float:
    rel = np.array([p is not None and p == t for p, t in zip(pred, truth)], np.float64)
    order = np.argsort(-np.asarray(conf, np.float64), kind="stable")
    rel = rel[order]
    n_obj = sum(t is not None for t in truth)
    if n_obj == 0:
        return 0.0
    prec = np.cumsum(rel) / (1 + np.arange(len(rel)))
    return float((prec * rel).sum() / n_obj)


def summarize(pred: list, conf: list[float], truth: list, accept: float, confirm: float) -> dict:
    """Indicateurs produit (ARCHITECTURE.md §9)."""
    pred, truth = list(pred), list(truth)
    conf = np.asarray(conf, np.float64)
    known = np.array([t is not None for t in truth])
    correct = np.array([p == t and t is not None for p, t in zip(pred, truth)])
    acc_auto = conf >= accept
    acc_any = conf >= confirm
    n_known = int(known.sum())
    n_distr = int((~known).sum())
    out = {
        "n_queries": len(truth),
        "n_known": n_known,
        "n_distractors": n_distr,
        "top1_accuracy": float(correct[known].mean()) if n_known else None,
        "gap": gap(pred, conf.tolist(), truth),
        # parmi les ajouts automatiques, combien sont faux (y compris distracteurs acceptés)
        "false_accept_rate_auto": float((acc_auto & ~correct).sum() / max(acc_auto.sum(), 1)),
        "auto_accept_coverage": float((acc_auto & known).sum() / max(n_known, 1)),
        "confirm_or_better_coverage": float((acc_any & known).sum() / max(n_known, 1)),
        # objets hors base correctement rejetés (pas d'ajout automatique)
        "distractor_rejection": float((~acc_auto & ~known).sum() / n_distr) if n_distr else None,
    }
    return out


def format_summary(name: str, s: dict) -> str:
    def pct(v):
        return "  n/a" if v is None else f"{100 * v:5.1f} %"
    return (f"{name}\n"
            f"  requêtes : {s['n_queries']} ({s['n_known']} connues, {s['n_distractors']} hors base)\n"
            f"  top-1 (œuvres connues)            {pct(s['top1_accuracy'])}\n"
            f"  GAP                               {pct(s['gap'])}\n"
            f"  fausses acceptations (auto)       {pct(s['false_accept_rate_auto'])}\n"
            f"  couverture ajout automatique      {pct(s['auto_accept_coverage'])}\n"
            f"  couverture auto + confirmation    {pct(s['confirm_or_better_coverage'])}\n"
            f"  rejet des objets hors base        {pct(s['distractor_rejection'])}")
