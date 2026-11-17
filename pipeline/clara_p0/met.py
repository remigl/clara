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


def sample_queries(qs: list, limit: int | None, seed: int = 0) -> list:
    """Échantillon aléatoire (reproductible) qui garde la proportion œuvres connues / hors base.
    Les fichiers du Met sont triés : prendre les N premières lignes ne donne que des œuvres connues."""
    if not limit or limit >= len(qs):
        return qs
    idx = np.random.default_rng(seed).choice(len(qs), limit, replace=False)
    return [qs[i] for i in sorted(idx)]


def balanced_queries(qs: list, n_known: int | None = None, n_distractors: int | None = 1000,
                     seed: int = 0, keep=None) -> list:
    """Toutes les requêtes connues (ou n_known) + n_distractors objets hors base, tirés au hasard.
    Le testset du Met contient ~20 fois plus de distracteurs que d'œuvres connues : sans équilibrage,
    un échantillon ne garde qu'une centaine d'œuvres connues. `keep(t)` filtre les œuvres connues."""
    rng = np.random.default_rng(seed)
    known = [q for q in qs if q[1] is not None and (keep is None or keep(q[1]))]
    distr = [q for q in qs if q[1] is None]
    if n_known is not None and n_known < len(known):
        known = [known[i] for i in sorted(rng.choice(len(known), n_known, replace=False))]
    if n_distractors is not None and n_distractors < len(distr):
        distr = [distr[i] for i in sorted(rng.choice(len(distr), n_distractors, replace=False))]
    return known + distr


def met_departments(ids, cache: Path) -> dict[str, str]:
    """« département|classification » de chaque objet via l'API ouverte du Met. Mis en cache."""
    import requests
    known = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [i for i in ids if i not in known]
    if todo:
        print(f"  API du Met : {len(todo)} objets à interroger...", flush=True)
    s = requests.Session()
    s.headers["User-Agent"] = "clara-p0/0.1 (https://github.com/remigl/clara)"
    for n, i in enumerate(todo, 1):
        try:
            r = s.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{i}", timeout=20)
            if r.status_code == 200:
                j = r.json()
                known[i] = f"{j.get('department', '')}|{j.get('classification', '')}"
            elif r.status_code == 404:
                known[i] = "?|?"
        except Exception:
            pass
        if n % 100 == 0:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(known))
            print(f"  API du Met : {n}/{len(todo)}", flush=True)
        time.sleep(0.02)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known))
    return known


def is_painting(meta: str | None) -> bool:
    """Tableau plat : classification « Paintings » ou département des peintures européennes."""
    if not meta:
        return False
    dept, cls = (meta.split("|") + [""])[:2]
    return cls.lower().startswith("painting") or dept == "European Paintings"


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
             cache_dir: str = "cache", out: str | None = None, limit: int | None = None,
             distractors: int | None = 1000, paintings: bool = False) -> dict:
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

    keep = None
    if paintings:
        ids = {t for sp in ("val", "test") for _, t in mp.queries(sp) if t is not None}
        meta = met_departments(ids, Path(cache_dir) / "met_objects.json")
        keep = lambda t: is_painting(meta.get(t))  # noqa: E731
        print(f"  peintures : {sum(keep(t) for t in ids)} œuvres connues sur {len(ids)}")

    def load(split):
        qs = balanced_queries(mp.queries(split), limit, distractors, keep=keep)
        print(f"  {split} : {sum(t is not None for _, t in qs)} œuvres connues + {sum(t is None for _, t in qs)} hors base")
        return ((read_image(mp.resolve(p)), t) for p, t in qs)

    print("Calibration sur valset…")
    val: RunOutput = run_queries(rec, load("val"))
    calibrator = val.fit_calibrator() if rerank else None
    if calibrator:
        calibrator.save(Path(cache_dir) / f"calibration_{embedder.model_id}.json")
    print("Mesure sur testset…")
    test = run_queries(rec, load("test"))
    res = {"model_id": embedder.model_id, "database": "mini" if mini else "full", "k": k, "paintings_only": paintings,
           **report("Met testset", test, calibrator, rerank)}
    if out:
        save_json(out, res)
    return res
