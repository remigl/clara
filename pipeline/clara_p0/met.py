"""Évaluation sur le Met Dataset (Ypsilantis et al., NeurIPS 2021 — http://cmp.felk.cvut.cz/met/).

Base : photos studio des œuvres du Met (une classe = une œuvre).
Requêtes : photos de visiteurs (avec MET_id) + distracteurs sans MET_id (hors base).
valset.json sert à calibrer la confiance, testset.json à mesurer.

Arborescence attendue (après `clara-p0 download-met`) :
  <root>/ground_truth/{mini_MET_database.json | MET_database.json, valset.json, testset.json}
  <root>/images/<chemins des JSON>   (ou directement <root>/<chemins>)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .embed import Embedder
from .evaluate import RunOutput, report, run_queries, save_json
from .imageio import read_image, resize_max
from .index import FlatIndex
from .recognize import Recognizer
from .verify import GeometricVerifier, LazyKeypoints


class MetPaths:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        hits = sorted(self.root.rglob("testset.json"))
        if not hits:
            raise FileNotFoundError(f"testset.json introuvable sous {self.root} (lance download-met)")
        self.gt = hits[0].parent
        self._bases = [self.root / "images", self.root, self.root / "images" / "images"]
        self._base: Path | None = None

    def resolve(self, rel: str) -> Path:
        if self._base is not None and (self._base / rel).exists():
            return self._base / rel
        for base in self._bases:
            if (base / rel).exists():
                self._base = base
                return base / rel
        raise FileNotFoundError(f"{rel} introuvable sous {self._bases}")

    def database(self, mini: bool) -> list[tuple[str, str]]:
        fn = "mini_MET_database.json" if mini else "MET_database.json"
        data = json.loads((self.gt / fn).read_text())
        return [(e["path"], str(e["id"])) for e in data]

    def queries(self, split: str) -> list[tuple[str, str | None]]:
        data = json.loads((self.gt / f"{split}set.json").read_text())
        return [(e["path"], str(e["MET_id"]) if "MET_id" in e else None) for e in data]


def embed_database(embedder: Embedder, mp: MetPaths, db: list[tuple[str, str]], cache: Path,
                   batch: int = 32, max_side: int = 512) -> FlatIndex:
    """Embeddings de la base avec cache disque reprenable (Ctrl+C sans perte)."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    done_vecs, done_n = [], 0
    if cache.exists():
        d = np.load(cache)
        done_vecs, done_n = [d["vectors"]], int(d["n"])
        print(f"  cache : {done_n}/{len(db)} images déjà encodées")
    t0 = time.perf_counter()
    imgs, vecs = [], list(done_vecs)
    n = done_n
    try:
        for i in range(done_n, len(db)):
            imgs.append(resize_max(read_image(mp.resolve(db[i][0])), max_side))
            if len(imgs) == batch or i == len(db) - 1:
                vecs.append(embedder.embed(imgs))
                n = i + 1
                imgs = []
                if (n - done_n) % (batch * 50) < batch:
                    rate = (time.perf_counter() - t0) / max(n - done_n, 1)
                    print(f"  base : {n}/{len(db)} ({rate * 1e3:.0f} ms/image, reste ~{rate * (len(db) - n) / 60:.0f} min)",
                          flush=True)
                    np.savez(cache, vectors=np.concatenate(vecs), n=n)
    finally:
        if vecs:
            np.savez(cache, vectors=np.concatenate(vecs), n=n)
    x = np.concatenate(vecs)
    index = FlatIndex(embedder.dim)
    counts: dict[str, int] = {}
    rows = []
    for _, aid in db[: len(x)]:
        rows.append((aid, counts.get(aid, 0), "full"))
        counts[aid] = counts.get(aid, 0) + 1
    index.add(x, rows)
    return index


def evaluate(embedder: Embedder, root: str, mini: bool = True, rerank: bool = True, k: int = 5,
             cache_dir: str = "cache", out: str | None = None, limit: int | None = None) -> dict:
    mp = MetPaths(root)
    db = mp.database(mini)
    print(f"Met Dataset : base {'mini' if mini else 'complète'} = {len(db)} images, modèle {embedder.model_id}")
    cache = Path(cache_dir) / f"met_{'mini' if mini else 'full'}_{embedder.model_id}.npz"
    index = embed_database(embedder, mp, db, cache)

    first_path: dict[str, str] = {}
    for p, aid in db:
        first_path.setdefault(aid, p)
    verifier = GeometricVerifier() if rerank else None
    kps = LazyKeypoints(lambda a: mp.resolve(first_path[a]) if a in first_path else None, verifier) if rerank else None
    rec = Recognizer(embedder, index, kps, verifier, k=k, rerank=rerank)

    def load(split):
        qs = mp.queries(split)
        if limit:
            qs = qs[:limit]
        return ((read_image(mp.resolve(p)), t) for p, t in qs)

    print("Calibration sur valset…")
    val: RunOutput = run_queries(rec, load("val"))
    calibrator = val.fit_calibrator() if rerank else None
    if calibrator:
        calibrator.save(Path(cache_dir) / f"calibration_{embedder.model_id}.json")
    print("Mesure sur testset…")
    test = run_queries(rec, load("test"))
    res = {"model_id": embedder.model_id, "database": "mini" if mini else "full", "k": k,
           **report("Met testset", test, calibrator, rerank)}
    if out:
        save_json(out, res)
    return res
