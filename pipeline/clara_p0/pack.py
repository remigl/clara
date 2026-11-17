"""Construction et lecture d'un pack musée (format décrit dans ARCHITECTURE.md §3.6).

<out>/<museum_id>-v<version>/
  manifest.json   catalog.db   vectors.bin   scales.bin   keypoints.bin   thumbs/<id>.webp
+ <museum_id>-v<version>.zip
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .embed import Embedder
from .imageio import read_image, resize_max, write_image
from .index import FlatIndex
from .records import Artwork, Museum, Source
from .verify import GeometricVerifier, KeypointStore
from .views import DEFAULT_REFERENCE_VIEWS, reference_views

SCHEMA = Path(__file__).resolve().parents[2] / "data" / "schema.sql"
UA = "clara-p0/0.1 (https://github.com/remigl/clara)"
PROGRESS_STATUSES = {"on_display", "unknown", "temporarily_absent"}


def catalog_schema() -> str:
    text = SCHEMA.read_text(encoding="utf-8")
    marker = text.index("-- user.db\n---")
    return text[: text.rfind("\n", 0, marker - 2)]


def user_schema() -> str:
    text = SCHEMA.read_text(encoding="utf-8")
    return text[text.index("-- user.db\n---"):]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- images

def _download(url: str, dest: Path, session, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    if not url.startswith(("http://", "https://")):  # chemin local (tests, données déjà téléchargées)
        src = Path(url.removeprefix("file://"))
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            return True
        return False
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
                return True
            if r.status_code in (404, 410):
                return False
        except Exception:  # réseau instable : on réessaie
            pass
        time.sleep(1.5 * (attempt + 1))
    return False


def download_images(artworks: list[Artwork], cache_dir: Path, workers: int = 4) -> dict[str, Path]:
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = UA
    todo = [a for a in artworks if a.image_url]
    paths: dict[str, Path] = {}

    def job(a: Artwork):
        dest = cache_dir / (hashlib.sha1(a.image_url.encode()).hexdigest() + ".img")
        return a.id, dest if _download(a.image_url, dest, s) else None

    with ThreadPoolExecutor(workers) as ex:
        for i, (aid, p) in enumerate(ex.map(job, todo), 1):
            if p:
                paths[aid] = p
            if i % 100 == 0:
                print(f"  images : {i}/{len(todo)}", flush=True)
    print(f"  images : {len(paths)}/{len(todo)} disponibles")
    return paths


# --------------------------------------------------------------------------- build

def build(museum: Museum, sources: list[Source], artworks: list[Artwork], embedder: Embedder,
          out_dir: str | Path, cache_dir: str | Path = "cache", version: int = 1,
          views=DEFAULT_REFERENCE_VIEWS, thumbs: bool = True, embed_restricted: bool = False,
          zip_pack: bool = True) -> Path:
    """Construit un pack. Par défaut (politique v1, ARCHITECTURE.md §3.2), seules les images « open »
    sont encodées et affichées ; les autres œuvres restent dans le catalogue, sans image."""
    t0 = time.perf_counter()
    # dédoublonnage par id
    uniq: dict[str, Artwork] = {}
    for a in artworks:
        uniq.setdefault(a.id, a)
    artworks = list(uniq.values())
    print(f"Pack {museum.id} v{version} : {len(artworks)} œuvres")

    pack_dir = Path(out_dir) / f"{museum.id}-v{version}"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    (pack_dir / "thumbs").mkdir(parents=True)

    eligible = [a for a in artworks if a.image_url and (embed_restricted or a.image_rights_class == "open")]
    img_paths = download_images(eligible, Path(cache_dir) / "images")

    verifier = GeometricVerifier()
    index = FlatIndex(embedder.dim)
    kps = KeypointStore()
    thumb_info: dict[str, tuple[str, int, int]] = {}
    batch_imgs, batch_rows = [], []
    for i, a in enumerate(eligible, 1):
        p = img_paths.get(a.id)
        if p is None:
            continue
        try:
            img = resize_max(read_image(p), 1024)
        except ValueError:
            continue
        for vi, (name, v) in enumerate(reference_views(img, views)):
            batch_imgs.append(v)
            batch_rows.append((a.id, vi, name))
        kps[a.id] = verifier.extract(img)
        if thumbs and a.image_rights_class == "open":
            t = resize_max(img, 512)
            rel = f"thumbs/{hashlib.sha1(a.id.encode()).hexdigest()[:16]}.webp"
            write_image(pack_dir / rel, t, quality=80)
            thumb_info[a.id] = (rel, t.shape[1], t.shape[0])
        if len(batch_imgs) >= 32:
            index.add(embedder.embed(batch_imgs), batch_rows)
            batch_imgs, batch_rows = [], []
        if i % 100 == 0:
            print(f"  encodage : {i}/{len(eligible)}", flush=True)
    if batch_imgs:
        index.add(embedder.embed(batch_imgs), batch_rows)

    index.save(pack_dir)
    (pack_dir / "rows.json").unlink()  # la correspondance lignes -> œuvres vit dans catalog.db
    offsets = kps.save(pack_dir / "keypoints.bin")

    db = sqlite3.connect(pack_dir / "catalog.db")
    db.executescript(catalog_schema())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    recognizable = {r[0] for r in index.rows}
    counted = [a for a in artworks if a.object_type == "painting" and a.exhibition_status in PROGRESS_STATUSES]
    counted_ids = {a.id for a in counted}

    db.execute("INSERT INTO pack VALUES (?,?,?,?,?,?,?)",
               (museum.id, version, now, embedder.model_id, embedder.dim, "int8", ""))
    m = asdict(museum)
    db.execute("""INSERT INTO museum (id,name,name_local,city,country,lat,lon,geofence_m,website,image_path,
                  wikidata_qid,data_quality,number_of_works,pack_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
               (m["id"], m["name"], m["name_local"], m["city"], m["country"], m["lat"], m["lon"], m["geofence_m"],
                m["website"], None, m["wikidata_qid"], m["data_quality"], len(counted), museum.id))
    for s in sources:
        db.execute("INSERT INTO source VALUES (?,?,?,?)", (s.id, s.name, s.url, s.license))

    colls: dict[str, str] = {}
    for a in artworks:
        if a.collection and a.collection not in colls:
            colls[a.collection] = f"{museum.id}:c{len(colls) + 1}"
    for name, cid in colls.items():
        n = sum(1 for a in counted if a.collection == name)
        db.execute("INSERT INTO collection VALUES (?,?,?,?)", (cid, museum.id, name, n))

    for a in artworks:
        if a.artist_qid or a.artist_display:
            art_id = a.artist_qid or f"name:{a.artist_display}"
            db.execute("INSERT OR IGNORE INTO artist VALUES (?,?,?,?)",
                       (art_id, a.artist_display or "Anonyme", a.artist_birth_year, a.artist_death_year))
        else:
            art_id = None
        db.execute("""INSERT INTO artwork (id,museum_id,collection_id,title,title_local,artist_id,artist_display,
                      date_display,date_start,date_end,description,inventory_number,medium,height_cm,width_cm,
                      object_type,room,floor,exhibition_status,status_source_id,status_checked_at,
                      counts_for_progress,wikidata_qid,source_id,source_url,updated_at)
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (a.id, museum.id, colls.get(a.collection), a.title, None, art_id, a.artist_display,
                    a.date_display, a.date_start, a.date_end, a.description, a.inventory_number, a.medium,
                    a.height_cm, a.width_cm, a.object_type, a.room, a.floor, a.exhibition_status,
                    a.source_id if a.exhibition_status != "unknown" else None, a.status_checked_at,
                    int(a.id in counted_ids), a.wikidata_qid, a.source_id, a.source_url, now))
        if a.image_url:
            t = thumb_info.get(a.id)
            db.execute("INSERT INTO artwork_image VALUES (?,?,?,?,?,?,?,?)",
                       (a.id, t[0] if t else None, t[1] if t else None, t[2] if t else None,
                        a.image_rights_uri or "", a.image_rights_class, a.credit_line, a.image_url))
    for row, (aid, vi, kind) in enumerate(index.rows):
        db.execute("INSERT INTO artwork_embedding VALUES (?,?,?,?)", (row, aid, vi, kind))
    for aid, (o, n) in offsets.items():
        db.execute("INSERT INTO artwork_keypoints VALUES (?,?,?)", (aid, o, n))
    db.execute("INSERT INTO artwork_fts(artwork_fts) VALUES ('rebuild')")
    db.commit()
    db.execute("VACUUM")
    db.close()

    files = sorted(p for p in pack_dir.rglob("*") if p.is_file())
    manifest = {
        "id": museum.id, "version": version, "built_at": now, "model_id": embedder.model_id,
        "embedding_dim": embedder.dim, "embedding_dtype": "int8", "reference_views": list(views),
        "data_quality": museum.data_quality,
        "counts": {"artworks": len(artworks), "recognizable": len(recognizable), "progress_denominator": len(counted),
                   "with_thumbnail": len(thumb_info), "vectors": len(index)},
        "files": {str(p.relative_to(pack_dir)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": _sha256(p)}
                  for p in files},
    }
    manifest["bytes_total"] = sum(f["bytes"] for f in manifest["files"].values())
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if zip_pack:
        zpath = pack_dir.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # images déjà compressées
            for p in sorted(pack_dir.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(pack_dir))
    c = manifest["counts"]
    print(f"  -> {pack_dir} : {c['recognizable']} œuvres reconnaissables / {c['artworks']}, "
          f"dénominateur progression {c['progress_denominator']}, {manifest['bytes_total'] / 1e6:.1f} Mo, "
          f"{time.perf_counter() - t0:.0f} s")
    return pack_dir


# --------------------------------------------------------------------------- lecture

class Pack:
    """Lecture d'un pack exactement comme l'app le fera (catalog.db + fichiers binaires)."""

    def __init__(self, pack_dir: str | Path):
        self.dir = Path(pack_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.db = sqlite3.connect(self.dir / "catalog.db")
        self.db.row_factory = sqlite3.Row
        dim = self.manifest["embedding_dim"]
        rows = [(r["artwork_id"], r["view_idx"], r["kind"])
                for r in self.db.execute("SELECT * FROM artwork_embedding ORDER BY row")]
        self.index = FlatIndex(dim)
        self.index.rows = rows
        self.index.q = np.fromfile(self.dir / "vectors.bin", np.int8).reshape(len(rows), dim)
        self.index.scale = np.fromfile(self.dir / "scales.bin", np.float32)
        offsets = {r["artwork_id"]: (r["offset"], r["length"]) for r in self.db.execute("SELECT * FROM artwork_keypoints")}
        self.keypoints = KeypointStore.load(self.dir / "keypoints.bin", offsets)

    def artwork(self, aid: str) -> dict | None:
        r = self.db.execute("SELECT * FROM artwork WHERE id = ?", (aid,)).fetchone()
        return dict(r) if r else None

    def search_text(self, q: str, limit: int = 10) -> list[dict]:
        sql = ("SELECT a.id, a.title, a.artist_display FROM artwork_fts f JOIN artwork a ON a.rowid = f.rowid "
               "WHERE artwork_fts MATCH ? LIMIT ?")
        return [dict(r) for r in self.db.execute(sql, (q, limit))]

    def verify_integrity(self) -> list[str]:
        bad = []
        for rel, info in self.manifest["files"].items():
            p = self.dir / rel
            if not p.exists() or _sha256(p) != info["sha256"]:
                bad.append(rel)
        return bad
