"""Comparaison rapide de variantes sur le Met Dataset, sans réencoder la base.

Réutilise le cache de `eval-met` (embeddings de la base) et compare, sur l'embedding seul :
- la façon de résumer DINOv2 : CLS, GeM des patchs, ou les deux ;
- avec / sans projection PCA-whitening 256-d (apprise sur la base, sans étiquettes) ;
- la taille de la base : tout le Met (~38 000 images) ou un « musée » de N œuvres (mode musée de Clara) ;
- toutes les œuvres, ou seulement les peintures (département via l'API ouverte du Met, optionnel).

Indicateurs : top-1, GAP, et couverture à 1 % de fausses acceptations (part des œuvres connues
reconnues sûrement quand on règle le seuil pour ne se tromper qu'une fois sur cent).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .embed import OnnxDinoEmbedder, Projection, l2n
from .imageio import read_image
from .met import MetPaths, balanced_queries, is_painting, met_departments
from .metrics import coverage_at_far, gap
from .views import query_views

def _query_cache(emb: OnnxDinoEmbedder, mp: MetPaths, qs, cache: Path):
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if int(d["n"]) == len(qs):
            return d["vecs"], d["nviews"]
    vecs = np.zeros((len(qs), 3, emb.raw_dim), np.float32)
    nviews = np.zeros(len(qs), np.int32)
    t0 = time.perf_counter()
    for i, (p, _) in enumerate(qs):
        views, _ = query_views(read_image(mp.resolve(p)))
        v = emb._raw([im for _, im in views])
        vecs[i, : len(v)] = v
        nviews[i] = len(v)
        if (i + 1) % 200 == 0:
            r = (time.perf_counter() - t0) / (i + 1)
            print(f"  requêtes : {i + 1}/{len(qs)} (reste ~{r * (len(qs) - i - 1) / 60:.0f} min)", flush=True)
    np.savez(cache, vecs=vecs, nviews=nviews, n=len(qs))
    return vecs, nviews


def run(model, root: str, mini: bool = True, limit: int | None = None, distractors: int = 1000, museum_sizes=(0, 2000),
        paintings: bool = False, cache_dir: str = "cache", out: str | None = None, seed: int = 0) -> list[dict]:
    emb = OnnxDinoEmbedder(model) if isinstance(model, (str, Path)) else model
    mp = MetPaths(root)
    db = mp.database(mini)
    db_cache = Path(cache_dir) / f"met_{'mini' if mini else 'full'}_{emb.model_id}.npz"
    if not db_cache.exists():
        raise SystemExit(f"Cache introuvable : {db_cache}. Lance d'abord `python -m clara_p0 eval-met --limit 50`.")
    d = np.load(db_cache)
    if int(d["n"]) < len(db):
        raise SystemExit("Le cache de la base est incomplet : relance eval-met jusqu'à la fin de l'encodage.")
    dbv = d["vectors"][: len(db)].astype(np.float32)
    h = dbv.shape[1] // 2
    db_parts = {"cls": l2n(dbv[:, :h]), "gem": l2n(dbv[:, h:])}

    labels = [aid for _, aid in db]
    classes = sorted(set(labels))
    cpos = {c: i for i, c in enumerate(classes)}
    group = np.array([cpos[a] for a in labels], np.int64)

    qs = balanced_queries(mp.queries("test"), limit, distractors, seed)
    truth = [t for _, t in qs]
    print(f"Sweep Met : base {len(db)} images / {len(classes)} œuvres ; {len(qs)} requêtes "
          f"({sum(t is not None for t in truth)} connues, {sum(t is None for t in truth)} hors base)")
    qv, nv = _query_cache(emb, mp, qs, Path(cache_dir) / f"met_queries_test_{emb.model_id}_{len(qs)}_s{seed}.npz")
    q_parts = {"cls": l2n(qv[..., :h]), "gem": l2n(qv[..., h:])}

    subsets = {"toutes œuvres": np.ones(len(qs), bool)}
    if paintings:
        meta = met_departments({t for t in truth if t is not None}, Path(cache_dir) / "met_objects.json")
        mask = np.array([t is None or is_painting(meta.get(t)) for t in truth])
        n_p = int(sum(1 for t, m in zip(truth, mask) if t is not None and m))
        ex = [(t, meta.get(t)) for t in truth if t is not None][:3]
        print(f"  exemples (id -> département|classification) : {ex}")
        if n_p >= 20:
            subsets["peintures"] = mask
            print(f"  {n_p} requêtes connues sont des peintures")
        else:
            print(f"  seulement {n_p} peintures identifiées : sous-ensemble ignoré")

    variants = {"CLS": ["cls"], "GeM": ["gem"], "CLS+GeM": ["cls", "gem"]}
    rng = np.random.default_rng(seed)
    rows = []
    for vname, parts in variants.items():
        D = l2n(np.concatenate([db_parts[p] for p in parts], axis=1))
        Q = l2n(np.concatenate([q_parts[p] for p in parts], axis=-1))
        for proj_name in ("brut", "PCAw-256"):
            if proj_name == "PCAw-256":
                pr = Projection.fit(D, dim=256, shrinkage=0.5)
                Dp, Qp = pr(D), pr(Q.reshape(-1, Q.shape[-1])).reshape(Q.shape[0], 3, -1)
            else:
                Dp, Qp = D, Q
            # meilleur score par œuvre pour chaque requête (max sur vues requête et images de référence)
            best = np.full((len(qs), len(classes)), -np.inf, np.float32)
            for i in range(len(qs)):
                s = (Qp[i, : nv[i]] @ Dp.T).max(0)
                np.maximum.at(best[i], group, s)
            for size in museum_sizes:
                if size and size < len(classes):
                    pred, score = [], []
                    for i, t in enumerate(truth):
                        sub = rng.choice(len(classes), size, replace=False)
                        if t is not None and cpos.get(t) is not None and cpos[t] not in sub:
                            sub[0] = cpos[t]
                        j = sub[np.argmax(best[i, sub])]
                        pred.append(classes[j]); score.append(best[i, j])
                    base = f"musée de {size}"
                else:
                    j = best.argmax(1)
                    pred = [classes[k] for k in j]
                    score = best[np.arange(len(qs)), j].tolist()
                    base = f"Met entier ({len(classes)})"
                score = np.array(score)
                for sname, m in subsets.items():
                    t_ = [truth[i] for i in np.where(m)[0]]
                    p_ = [pred[i] for i in np.where(m)[0]]
                    s_ = score[m]
                    known = np.array([t is not None for t in t_])
                    correct = np.array([p == t for p, t in zip(p_, t_)]) & known
                    rows.append({"variante": vname, "projection": proj_name, "base": base, "sous-ensemble": sname,
                                 "n_connues": int(known.sum()),
                                 "top1": float(correct[known].mean()) if known.any() else None,
                                 "gap": gap(p_, s_.tolist(), [t if t is not None else None for t in t_]),
                                 "couverture_far1": coverage_at_far(s_, correct, known, 0.01)})
    print(f"\n{'variante':9} {'proj.':9} {'base':22} {'sous-ensemble':14} {'top-1':>7} {'GAP':>7} {'sûr@1%':>7}")
    for r in rows:
        print(f"{r['variante']:9} {r['projection']:9} {r['base']:22} {r['sous-ensemble']:14} "
              f"{100 * r['top1']:6.1f}% {100 * r['gap']:6.1f}% {100 * r['couverture_far1']:6.1f}%")
    print("\nsûr@1% = part des œuvres reconnues avec un seuil réglé pour <= 1 % d'erreurs (embedding seul).")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows
