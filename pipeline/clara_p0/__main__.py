"""Ligne de commande de la phase P0 de Clara.

  python -m clara_p0 selftest                      test complet hors ligne (données synthétiques)
  python -m clara_p0 download-model                DINOv2-small ONNX (Apache 2.0) -> models/
  python -m clara_p0 download-met                  Met Dataset (mini) -> data/met
  python -m clara_p0 eval-met                      précision sur les photos de visiteurs du Met
  python -m clara_p0 sweep-met --paintings          variantes d'empreinte, mode musée, peintures seules
  python -m clara_p0 fit-projection                PCA-whitening 256-d appris sur la base Met
  python -m clara_p0 build-smk                     pack SMK (Copenhague)
  python -m clara_p0 build-wikidata --qid Q23402 --id fr-orsay
  python -m clara_p0 recognize --pack packs/dk-smk-v1 photo.jpg
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

MODELS = {
    "dinov2-small": "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx",
    "dinov2-small-int8": "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model_int8.onnx",
}
MET_BASE = "http://ptak.felk.cvut.cz/met/dataset/"
MET_FILES = {  # nom -> (taille annoncée, dossier d'extraction relatif à root)
    "ground_truth.tar.gz": ("3 Mo", "."),
    "test_met.tar.gz": ("32 Mo", "images"),
    "test_other.tar.gz": ("1 Go", "images"),
    "test_noart.tar.gz": ("1 Go", "images"),
    "small_MET.tar.gz": ("3 Go", "images"),
    "MET.tar.gz": ("28 Go", "images"),
}
DEFAULT_MODEL = Path("models/dinov2-small.onnx")


def _fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  déjà présent : {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "clara-p0/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0, last = 0, time.time(), 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                pct = f"{100 * done / total:5.1f} %" if total else ""
                print(f"\r  {dest.name} {done / 1e6:8.0f} Mo {pct} {done / 1e6 / max(time.time() - t0, 1e-3):5.1f} Mo/s",
                      end="", flush=True)
    tmp.replace(dest)
    print(f"\r  {dest.name} : {done / 1e6:.0f} Mo téléchargés" + " " * 20)


def _model_arg(a) -> str:
    if a.model in (None, "auto"):
        if DEFAULT_MODEL.exists():
            return str(DEFAULT_MODEL)
        print("Aucun modèle trouvé : lance d'abord `python -m clara_p0 download-model`, "
              "ou --model thumb pour la référence triviale.", file=sys.stderr)
        sys.exit(2)
    return a.model


def _embedder(a):
    from .embed import make_embedder
    return make_embedder(_model_arg(a), getattr(a, "projection", None))


def cmd_selftest(a):
    from . import selftest
    from .embed import make_embedder
    emb = make_embedder(None if a.model in (None, "thumb") else _model_arg(a), a.projection)
    selftest.run(emb, n=a.n, n_distractors=a.distractors, strength=a.strength, rerank=not a.no_rerank)


def cmd_download_model(a):
    _fetch(MODELS[a.variant], Path(a.out) / f"{a.variant}.onnx")
    print("Licence : Apache 2.0 (DINOv2, Meta AI ; export ONNX onnx-community).")


def cmd_download_met(a):
    root = Path(a.root)
    names = ["ground_truth.tar.gz", "test_met.tar.gz", "test_other.tar.gz", "test_noart.tar.gz",
             "MET.tar.gz" if a.full else "small_MET.tar.gz"]
    print("Met Dataset (Ypsilantis et al., NeurIPS 2021). À télécharger : "
          + ", ".join(f"{n} ({MET_FILES[n][0]})" for n in names))
    for n in names:
        tgz = root / "archives" / n
        _fetch(MET_BASE + n, tgz)
        stamp = root / "archives" / (n + ".extracted")
        if stamp.exists():
            continue
        print(f"  extraction de {n}…", flush=True)
        dest = root / MET_FILES[n][1]
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12
                t.extractall(dest)
        stamp.touch()
    print(f"OK : {root}")


def cmd_eval_met(a):
    from . import met
    met.evaluate(_embedder(a), a.root, mini=not a.full, rerank=not a.no_rerank, k=a.k,
                 cache_dir=a.cache, out=a.out, limit=a.limit)


def cmd_sweep_met(a):
    from . import sweep
    sizes = [int(x) for x in a.museum_sizes.split(",")]
    sweep.run(_model_arg(a), a.root, mini=not a.full, limit=a.limit, museum_sizes=sizes,
              paintings=a.paintings, cache_dir=a.cache, out=a.out)


def cmd_fit_projection(a):
    import numpy as np
    from .embed import Projection, make_embedder
    from .met import MetPaths, embed_database
    emb = make_embedder(_model_arg(a))  # sans projection
    mp = MetPaths(a.root)
    db = mp.database(mini=not a.full)
    rng = np.random.default_rng(0)
    sample = [db[i] for i in sorted(rng.choice(len(db), min(a.sample, len(db)), replace=False))]
    cache = Path(a.cache) / f"projsample_{emb.model_id}_{len(sample)}.npz"
    idx = embed_database(emb, mp, sample, cache)
    x = idx.q.astype(np.float32) * idx.scale[:, None]
    proj = Projection.fit(x, dim=a.dim, shrinkage=a.shrinkage)
    proj.save(a.out)
    print(f"Projection {x.shape[1]} -> {a.dim} enregistrée : {a.out}")


def cmd_build_smk(a):
    from . import pack
    from .sources import smk
    raw_dir = Path(a.cache) / "smk"
    items = smk.load_raw(raw_dir) if a.offline_raw else smk.fetch(raw_dir, max_items=a.max_items)
    arts = smk.select(items, paintings_only=not a.all_types, on_display_only=not a.include_not_on_display)
    print(f"SMK : {len(items)} notices, {len(arts)} retenues")
    pack.build(smk.MUSEUM, [smk.SOURCE], arts, _embedder(a), a.out, a.cache, version=a.version)


def cmd_build_wikidata(a):
    from . import pack
    from .sources import wikidata
    raw_dir = Path(a.cache) / f"wikidata_{a.id}"
    museum = wikidata.museum(a.qid, a.id, lang=a.lang)
    rows = wikidata.load_raw(raw_dir) if a.offline_raw else wikidata.fetch(a.qid, raw_dir, lang=a.lang)
    arts = wikidata.select(rows, a.id)
    print(f"Wikidata {a.qid} ({museum.name}) : {len(arts)} peintures")
    pack.build(museum, [wikidata.SOURCE], arts, _embedder(a), a.out, a.cache, version=a.version)


def cmd_recognize(a):
    from .imageio import read_image
    from .pack import Pack
    from .recognize import Calibrator, Recognizer
    p = Pack(a.pack)
    emb = _embedder(a)
    if emb.model_id != p.manifest["model_id"]:
        print(f"Attention : pack construit avec {p.manifest['model_id']}, modèle courant {emb.model_id}", file=sys.stderr)
    cal = Calibrator.load(a.calibration) if a.calibration else None
    rec = Recognizer(emb, p.index, p.keypoints, calibrator=cal)
    for path in a.images:
        r = rec.recognize(read_image(path))
        print(f"\n{path} -> {r.decision} (confiance {r.confidence:.2f}, {sum(r.ms.values()):.0f} ms)")
        for c in r.candidates:
            art = p.artwork(c.artwork_id) or {}
            print(f"  {c.cos:5.3f}  {c.inliers:4d} inliers  {art.get('title', c.artwork_id)} — "
                  f"{art.get('artist_display', '')}  [{art.get('room') or art.get('exhibition_status', '')}]")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="clara_p0", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_opts(p, default="auto"):
        p.add_argument("--model", default=default, help="chemin .onnx, 'thumb' (référence triviale) ou auto")
        p.add_argument("--projection", help="fichier .npz de fit-projection")

    p = sub.add_parser("selftest"); model_opts(p, default="thumb")
    p.add_argument("--n", type=int, default=300); p.add_argument("--distractors", type=int, default=100)
    p.add_argument("--strength", type=float, default=1.0, help="intensité des dégradations (0.3 doux, 1 réaliste)")
    p.add_argument("--no-rerank", action="store_true"); p.set_defaults(f=cmd_selftest)

    p = sub.add_parser("download-model")
    p.add_argument("--variant", choices=list(MODELS), default="dinov2-small"); p.add_argument("--out", default="models")
    p.set_defaults(f=cmd_download_model)

    p = sub.add_parser("download-met")
    p.add_argument("--root", default="data/met"); p.add_argument("--full", action="store_true", help="base complète (28 Go)")
    p.set_defaults(f=cmd_download_met)

    p = sub.add_parser("eval-met"); model_opts(p)
    p.add_argument("--root", default="data/met"); p.add_argument("--full", action="store_true")
    p.add_argument("--no-rerank", action="store_true"); p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, help="limiter le nombre de requêtes (essai rapide)")
    p.add_argument("--cache", default="cache"); p.add_argument("--out", default="results/met.json")
    p.set_defaults(f=cmd_eval_met)

    p = sub.add_parser("sweep-met"); p.add_argument("--model", default="auto")
    p.add_argument("--root", default="data/met"); p.add_argument("--full", action="store_true")
    p.add_argument("--limit", type=int, default=2000, help="requêtes tirées au hasard dans testset")
    p.add_argument("--museum-sizes", default="0,2000", help="0 = tout le Met ; N = musée simulé de N œuvres")
    p.add_argument("--paintings", action="store_true", help="résultats séparés pour les peintures (API du Met)")
    p.add_argument("--cache", default="cache"); p.add_argument("--out", default="results/sweep.json")
    p.set_defaults(f=cmd_sweep_met)

    p = sub.add_parser("fit-projection"); p.add_argument("--model", default="auto")
    p.add_argument("--root", default="data/met"); p.add_argument("--full", action="store_true")
    p.add_argument("--sample", type=int, default=20000); p.add_argument("--dim", type=int, default=256)
    p.add_argument("--shrinkage", type=float, default=0.5); p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="models/projection-256.npz"); p.set_defaults(f=cmd_fit_projection)

    for name, f in (("build-smk", cmd_build_smk), ("build-wikidata", cmd_build_wikidata)):
        p = sub.add_parser(name); model_opts(p)
        p.add_argument("--out", default="packs"); p.add_argument("--cache", default="cache")
        p.add_argument("--version", type=int, default=1)
        p.add_argument("--offline-raw", action="store_true", help="réutiliser les métadonnées déjà téléchargées")
        if name == "build-smk":
            p.add_argument("--all-types", action="store_true", help="pas seulement les peintures")
            p.add_argument("--include-not-on-display", action="store_true")
            p.add_argument("--max-items", type=int)
        else:
            p.add_argument("--qid", required=True); p.add_argument("--id", required=True)
            p.add_argument("--lang", default="fr")
        p.set_defaults(f=f)

    p = sub.add_parser("recognize"); model_opts(p)
    p.add_argument("--pack", required=True); p.add_argument("--calibration")
    p.add_argument("images", nargs="+"); p.set_defaults(f=cmd_recognize)

    a = ap.parse_args(argv)
    if getattr(a, "out", None) and a.cmd == "eval-met":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    a.f(a)


if __name__ == "__main__":
    main()
