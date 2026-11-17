"""Test de bout en bout sans aucune donnée externe.

Base : N fausses peintures, dont des « séries » (variantes proches, comme les Nymphéas).
Requêtes : une photo de musée simulée par œuvre + des distracteurs (œuvres hors base).
La moitié des requêtes sert à calibrer la confiance, l'autre moitié à mesurer.
"""
from __future__ import annotations

import numpy as np

from .embed import Embedder
from .evaluate import build_reference, report, run_queries
from .recognize import Recognizer
from .synth import museum_photo, procedural_painting
from .verify import GeometricVerifier


def make_corpus(n: int, series_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    refs: dict[str, np.ndarray] = {}
    i = 0
    while len(refs) < n:
        base = procedural_painting(rng)
        refs[f"w{i:05d}"] = base
        i += 1
        if rng.random() < series_frac:  # série : 1 à 3 variantes proches
            for _ in range(int(rng.integers(1, 4))):
                if len(refs) >= n:
                    break
                refs[f"w{i:05d}"] = procedural_painting(rng, base=base)
                i += 1
    return refs


def run(embedder: Embedder, n: int = 300, n_distractors: int = 100, series_frac: float = 0.3,
        strength: float = 1.0, seed: int = 0, rerank: bool = True) -> dict:
    print(f"Selftest : {n} œuvres, {n_distractors} distracteurs, modèle {embedder.model_id}")
    refs = make_corpus(n, series_frac, seed)
    verifier = GeometricVerifier() if rerank else None
    index, kps = build_reference(embedder, refs.items(), verifier=verifier, log_every=0)
    rng = np.random.default_rng(seed + 1)
    queries = [(museum_photo(img, rng, strength=strength), aid) for aid, img in refs.items()]
    queries += [(museum_photo(procedural_painting(rng), rng, strength=strength), None)
                for _ in range(n_distractors)]
    order = rng.permutation(len(queries))
    queries = [queries[i] for i in order]
    half = len(queries) // 2
    rec = Recognizer(embedder, index, kps, verifier, rerank=rerank)
    calib_run = run_queries(rec, queries[:half], log_every=0)
    calibrator = calib_run.fit_calibrator() if rerank else None
    test_run = run_queries(rec, queries[half:], log_every=0)
    return report("Selftest (moitié test)", test_run, calibrator, rerank)
