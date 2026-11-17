"""Évaluation générique : base de référence + requêtes -> métriques, avec et sans reranking."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .embed import Embedder
from .index import FlatIndex
from .metrics import format_summary, summarize
from .recognize import ACCEPT, CONFIRM, Calibrator, Recognizer, Result
from .verify import GeometricVerifier, KeypointStore
from .views import reference_views


def build_reference(embedder: Embedder, items: Iterable[tuple[str, np.ndarray]],
                    views=("full", "crop_center"), verifier: GeometricVerifier | None = None,
                    batch: int = 32, log_every: int = 500) -> tuple[FlatIndex, KeypointStore | None]:
    index = FlatIndex(embedder.dim)
    kps = KeypointStore() if verifier else None
    buf_imgs, buf_rows = [], []
    t0, n = time.perf_counter(), 0

    def flush():
        if buf_imgs:
            index.add(embedder.embed(buf_imgs), list(buf_rows))
            buf_imgs.clear(); buf_rows.clear()

    for aid, img in items:
        for vi, (name, v) in enumerate(reference_views(img, views)):
            buf_imgs.append(v); buf_rows.append((aid, vi, name))
        if kps is not None:
            kps[aid] = verifier.extract(img)
        n += 1
        if len(buf_imgs) >= batch:
            flush()
        if log_every and n % log_every == 0:
            print(f"  référence : {n} œuvres ({(time.perf_counter() - t0) / n * 1e3:.0f} ms/œuvre)", flush=True)
    flush()
    return index, kps


@dataclass
class RunOutput:
    truth: list
    results: list[Result]

    def summary(self, calibrator: Calibrator | None = None, use_rerank: bool = True) -> dict:
        if use_rerank:
            conf = [calibrator.predict(r.features) if calibrator and r.candidates else r.confidence
                    for r in self.results]
            pred = [r.best for r in self.results]
        else:
            # sans reranking : prédiction cosinus, confiance = cos1 recalé linéairement (classement seul compte pour GAP)
            pred = [r.cos_best for r in self.results]
            conf = [r.features["cos1"] for r in self.results]
        s = summarize(pred, conf, self.truth, ACCEPT, CONFIRM)
        if not use_rerank:
            s["false_accept_rate_auto"] = s["auto_accept_coverage"] = None
            s["confirm_or_better_coverage"] = s["distractor_rejection"] = None
        lat = np.array([sum(r.ms.values()) for r in self.results]) if self.results else np.zeros(1)
        s["latency_ms_p50"] = float(np.percentile(lat, 50))
        s["latency_ms_p95"] = float(np.percentile(lat, 95))
        return s

    def fit_calibrator(self) -> Calibrator:
        feats = [r.features for r in self.results if r.candidates]
        ok = [r.best == t and t is not None for r, t in zip(self.results, self.truth) if r.candidates]
        return Calibrator.fit(feats, ok)


def run_queries(rec: Recognizer, queries: Iterable[tuple[np.ndarray, str | None]],
                log_every: int = 100) -> RunOutput:
    truth, results = [], []
    t0 = time.perf_counter()
    for i, (img, t) in enumerate(queries, 1):
        results.append(rec.recognize(img))
        truth.append(t)
        if log_every and i % log_every == 0:
            print(f"  requêtes : {i} ({(time.perf_counter() - t0) / i * 1e3:.0f} ms/requête)", flush=True)
    return RunOutput(truth, results)


def report(name: str, run: RunOutput, calibrator: Calibrator | None, rerank: bool) -> dict:
    out = {}
    s_cos = run.summary(use_rerank=False)
    out["cosine_only"] = s_cos
    print(format_summary(f"{name} — embedding seul", s_cos))
    if rerank:
        s_full = run.summary(calibrator, use_rerank=True)
        out["with_rerank"] = s_full
        print(format_summary(f"{name} — embedding + vérification géométrique + confiance", s_full))
        print(f"  latence p50 / p95 : {s_full['latency_ms_p50']:.0f} / {s_full['latency_ms_p95']:.0f} ms (CPU de ce PC)")
    return out


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
