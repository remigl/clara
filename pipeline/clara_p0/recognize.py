"""Chaîne de reconnaissance complète : vues -> embedding -> top-k -> reranking -> confiance -> décision.

C'est la référence Python de ce que fera clara_core sur l'appareil.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embed import Embedder
from .index import FlatIndex
from .verify import GeometricVerifier, KeypointStore
from .views import query_views

ACCEPT = 0.90
CONFIRM = 0.50

FEATURES = ["cos1", "cos_margin", "inl1", "inl_margin", "det"]


@dataclass
class Candidate:
    artwork_id: str
    cos: float
    inliers: int = -1


@dataclass
class Result:
    candidates: list[Candidate]
    features: dict[str, float]
    confidence: float
    decision: str  # auto_accept | confirm | no_match
    ms: dict[str, float] = field(default_factory=dict)
    cos_best: str | None = None   # meilleur candidat avant reranking (pour comparer)

    @property
    def best(self) -> str | None:
        return self.candidates[0].artwork_id if self.candidates else None


class Calibrator:
    """Régression logistique sur les features -> probabilité d'être correct.

    Poids par défaut raisonnables avant calibration ; `fit` les apprend sur un jeu de validation.
    """

    def __init__(self, w: np.ndarray | None = None, b: float | None = None,
                 mu: np.ndarray | None = None, sd: np.ndarray | None = None):
        # défaut : cos et inliers dominent (valeurs choisies à la main, à remplacer par fit)
        self.w = w if w is not None else np.array([6.0, 25.0, 0.08, 0.06, 0.5])
        self.b = b if b is not None else -6.5
        self.mu = mu if mu is not None else np.zeros(len(FEATURES))
        self.sd = sd if sd is not None else np.ones(len(FEATURES))

    def predict(self, f: dict[str, float]) -> float:
        x = (np.array([f[k] for k in FEATURES]) - self.mu) / self.sd
        return float(1 / (1 + np.exp(-(x @ self.w + self.b))))

    @classmethod
    def fit(cls, feats: list[dict[str, float]], correct: list[bool], l2: float = 1e-2,
            iters: int = 3000, lr: float = 0.5) -> "Calibrator":
        x = np.array([[f[k] for k in FEATURES] for f in feats], np.float64)
        y = np.array(correct, np.float64)
        mu, sd = x.mean(0), x.std(0) + 1e-9
        xs = (x - mu) / sd
        w, b = np.zeros(x.shape[1]), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(xs @ w + b)))
            g = p - y
            w -= lr * (xs.T @ g / len(y) + l2 * w)
            b -= lr * g.mean()
        return cls(w, b, mu, sd)

    def save(self, path):
        Path(path).write_text(json.dumps({"features": FEATURES, "w": self.w.tolist(), "b": self.b,
                                          "mu": self.mu.tolist(), "sd": self.sd.tolist()}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["w"]), d["b"], np.array(d["mu"]), np.array(d["sd"]))


class Recognizer:
    def __init__(self, embedder: Embedder, index: FlatIndex, keypoints: KeypointStore | None = None,
                 verifier: GeometricVerifier | None = None, calibrator: Calibrator | None = None,
                 k: int = 5, rerank: bool = True):
        self.embedder = embedder
        self.index = index
        self.keypoints = keypoints
        self.verifier = verifier or (GeometricVerifier() if keypoints is not None else None)
        self.calibrator = calibrator or Calibrator()
        self.k = k
        self.rerank = rerank and keypoints is not None

    def recognize(self, img: np.ndarray) -> Result:
        t0 = time.perf_counter()
        views, det = query_views(img)
        vecs = self.embedder.embed([v for _, v in views])
        t1 = time.perf_counter()
        # fusion des vues requête : max par œuvre
        merged: dict[str, float] = {}
        for top in self.index.search(vecs, k=self.k):
            for aid, s in top:
                merged[aid] = max(merged.get(aid, -1.0), s)
        cands = [Candidate(a, s) for a, s in sorted(merged.items(), key=lambda x: -x[1])[: self.k]]
        t2 = time.perf_counter()
        cos_best = cands[0].artwork_id if cands else None
        if self.rerank and cands:
            q_kp = self.verifier.extract(views[0][1])
            for c in cands:
                ref = self.keypoints.get(c.artwork_id)
                c.inliers = self.verifier.inliers(q_kp, ref) if ref is not None else 0
            # tri : inliers d'abord quand la géométrie est concluante, sinon cosinus
            cands.sort(key=lambda c: (c.inliers >= 15, c.inliers if c.inliers >= 15 else 0, c.cos), reverse=True)
        t3 = time.perf_counter()
        f = self._features(cands, det)
        p = self.calibrator.predict(f) if cands else 0.0
        decision = "auto_accept" if p >= ACCEPT else "confirm" if p >= CONFIRM else "no_match"
        return Result(cands, f, p, decision,
                      {"embed": (t1 - t0) * 1e3, "search": (t2 - t1) * 1e3, "rerank": (t3 - t2) * 1e3},
                      cos_best)

    def _features(self, c: list[Candidate], det: float) -> dict[str, float]:
        if not c:
            return {k: 0.0 for k in FEATURES}
        cos2 = c[1].cos if len(c) > 1 else 0.0
        i1 = max(c[0].inliers, 0)
        i2 = max((x.inliers for x in c[1:]), default=0)
        return {"cos1": c[0].cos, "cos_margin": c[0].cos - max((x.cos for x in c[1:]), default=cos2),
                "inl1": float(i1), "inl_margin": float(i1 - max(i2, 0)), "det": det}


def temporal_vote(results: list[Result], need: int = 3, min_conf: float = CONFIRM) -> str | None:
    """Vote sur les dernières images de la visée : une œuvre validée si `need` images s'accordent."""
    votes = Counter(r.best for r in results if r.best and r.confidence >= min_conf)
    if not votes:
        return None
    aid, n = votes.most_common(1)[0]
    return aid if n >= need else None
