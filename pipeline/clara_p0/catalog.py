"""Catalogue des musées : construction de tous les packs listés dans museums.json et assemblage de
catalog.json, le fichier que l'app télécharge pour proposer les musées.

catalog.json :
{
  "format": 1, "model_id": "...", "built_at": "...",
  "packs": [{"id", "name", "city", "country", "country_name", "lat", "lon", "version", "works",
             "recognizable", "data_quality", "file", "bytes", "sha256"}]
}
Tous les packs d'un catalogue doivent avoir été construits avec le même modèle (même model_id) que l'app.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PACK_FORMAT = 3  # à incrémenter si le format des packs change sans que le code ci-dessous change
# Fichiers dont dépend le contenu d'un pack (l'évaluation, la reconnaissance PC… n'en font pas partie)
RECIPE_FILES = ["pack.py", "records.py", "embed.py", "index.py", "views.py", "imageio.py", "verify.py",
                "catalog.py", "__main__.py", "sources/*.py"]
ROOT = Path(__file__).resolve().parent


def code_hash() -> str:
    h = hashlib.sha256()
    files = sorted({p for pat in RECIPE_FILES for p in ROOT.glob(pat)})
    files.append(ROOT.parents[1] / "data" / "schema.sql")
    for f in files:
        h.update(f.relative_to(ROOT.parents[1]).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def recipe(entry: dict, model: str, code: str | None = None) -> str:
    """Empreinte de fabrication d'un pack : même empreinte = même pack, inutile de le reconstruire."""
    e = {k: v for k, v in entry.items() if not k.startswith("_")}
    payload = json.dumps({"entry": e, "model": model, "format": PACK_FORMAT, "code": code or code_hash()},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def plan(registry: str, model: str, previous: str | None = None, force: str = "",
         max_age_days: float = 60) -> dict:
    """Musées à construire et musées à reprendre tels quels de la release précédente.

    Repris : même empreinte de fabrication et pack de moins de max_age_days jours (les données des musées
    évoluent : un rafraîchissement régulier reste utile). force : « all » ou « id1,id2 »."""
    entries = load_registry(registry)
    code = code_hash()
    prev = {}
    if previous and Path(previous).exists():
        prev = {p["id"]: p for p in json.loads(Path(previous).read_text(encoding="utf-8")).get("packs", [])}
    forced = {e["id"] for e in entries} if force.strip() == "all" else {f.strip() for f in force.split(",") if f.strip()}
    now = datetime.now(timezone.utc)
    build, reuse = [], []
    for e in entries:
        r = recipe(e, model, code)
        p = prev.get(e["id"])
        fresh = False
        if p and p.get("built_at"):
            age = now - datetime.strptime(p["built_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            fresh = age.days < max_age_days
        if e["id"] not in forced and p and p.get("recipe") == r and fresh:
            reuse.append({"id": e["id"], "file": p["file"]})
        else:
            build.append(e["id"])
    return {"build": build, "reuse": reuse, "code": code}


def load_registry(path: str | Path) -> list[dict]:
    reg = json.loads(Path(path).read_text(encoding="utf-8"))
    return reg["museums"] if isinstance(reg, dict) else reg


def _notice(msg: str) -> None:
    """Visible dans les annotations GitHub (vérification des musées sans lire tout le journal)."""
    print(msg, flush=True)
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title=pack::{msg}", flush=True)


MAX_RAW_AGE_DAYS = float(os.environ.get("CLARA_MAX_RAW_AGE_DAYS", "7"))


def _fresh(path: Path) -> bool:
    """Données brutes récentes : réutilisées plutôt que retéléchargées (constructions bien plus rapides)."""
    import time
    return path.exists() and (time.time() - path.stat().st_mtime) < MAX_RAW_AGE_DAYS * 86400


def build_one(entry: dict, emb, out: str, cache: str, version: int = 1, model: str = "dinov2-small") -> Path:
    from . import pack
    from .__main__ import build_wikidata_pack

    from .sources import knowledge, wikidata

    qid = entry.get("qid")
    if not qid and entry.get("search"):
        try:
            qid = wikidata.resolve_qid(entry["search"], entry.get("search_lang", "de"))
        except Exception as e:
            if entry["source"] == "wikidata":
                raise
            print(f"Identifiant Wikidata introuvable ({e}) : pack sans croisement")  # SMK : le pack reste utilisable
    if entry["source"] == "smk":
        from .sources import smk
        raw_dir = Path(cache) / "smk"
        try:
            if _fresh(raw_dir / "raw.jsonl"):
                print("Notices SMK récentes en cache : réutilisées")
                items = smk.load_raw(raw_dir)
            else:
                items = smk.fetch(raw_dir)
        except Exception:
            if not (raw_dir / "raw.jsonl").exists():
                raise
            print("API SMK indisponible : notices en cache réutilisées")
            items = smk.load_raw(raw_dir)
        arts = smk.select(items, paintings_only=True, on_display_only=True)
        if qid:  # croisement avec Wikidata par numéro d'inventaire : artistes, contexte, provenance, Wikipédia
            try:
                wd_dir = Path(cache) / f"wikidata_{entry['id']}"
                raw = (wikidata.load_raw(wd_dir) if _fresh(wd_dir / "raw.json")
                       else wikidata.fetch(qid, wd_dir, match=entry.get("match", "collection")))
                know = knowledge.gather(raw, Path(cache) / "knowledge.json")
                n = wikidata.enrich_by_inventory(arts, raw, know, entry["id"])
                _notice(f"{entry['id']} : {n}/{len(arts)} œuvres croisées avec Wikidata")
            except Exception as e:
                print(f"Croisement Wikidata impossible ({e.__class__.__name__}: {e}) : pack sans enrichissement")
        d = pack.build(smk.MUSEUM, [smk.SOURCE, wikidata.SOURCE], arts, emb, out, cache, version=version,
                       keypoints=False, recipe=recipe(entry, model))
    elif entry["source"] == "wikidata":
        d = build_wikidata_pack(qid, entry["id"], emb, out, cache, version=version,
                                lang=entry.get("lang", "fr"), city=entry.get("city"),
                                country_name=entry.get("country_name"), max_images=entry.get("max_images", 4000),
                                keypoints=False, country=entry.get("country"), match=entry.get("match", "collection"),
                                types=tuple(entry.get("types", ["painting"])),
                                offline_raw=_fresh(Path(cache) / f"wikidata_{entry['id']}" / "raw.json"),
                                recipe=recipe(entry, model))
    else:
        raise ValueError(f"source inconnue : {entry['source']}")
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    c = m["counts"]
    _notice(f"{entry['id']} = {m['museum']['name']} ({m['museum']['city']}, {m['museum']['country']}) : "
            f"{c['artworks']} œuvres, {c['recognizable']} reconnaissables, {m['bytes_total'] / 1e6:.0f} Mo")
    return d


def build_all(registry: str, emb, out: str = "packs", cache: str = "cache", only: str | None = None,
              version: int = 1) -> None:
    entries = [e for e in load_registry(registry) if not only or e["id"] == only]
    if only and not entries:
        raise SystemExit(f"musée inconnu dans {registry} : {only}")
    for e in entries:
        build_one(e, emb, out, cache, version)
    if not only:
        merge(out, registry)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge(packs_dir: str, registry: str | None = None) -> Path:
    """Assemble catalog.json à partir des manifestes et des zips présents dans packs_dir (cherchés récursivement,
    pour accepter les artefacts téléchargés de jobs parallèles)."""
    root = Path(packs_dir)
    reg = {e["id"]: e for e in load_registry(registry)} if registry and Path(registry).exists() else {}
    best: dict[str, dict] = {}
    for man in root.rglob("manifest.json"):
        m = json.loads(man.read_text(encoding="utf-8"))
        z = man.parent.with_suffix(".zip")
        if not z.exists():
            hits = list(root.rglob(f"{man.parent.name}.zip"))
            if not hits:
                continue
            z = hits[0]
        if m["id"] in best and best[m["id"]]["version"] >= m["version"]:
            continue
        mu = m.get("museum", {})
        e = reg.get(m["id"], {})
        best[m["id"]] = {
            "id": m["id"], "name": mu.get("name_local") or mu.get("name") or m["id"],
            "name_en": mu.get("name"), "city": e.get("city") or mu.get("city", ""),
            "country": mu.get("country") or e.get("country", ""),
            "country_name": e.get("country_name") or mu.get("country_name") or mu.get("country", ""),
            "lat": mu.get("lat"), "lon": mu.get("lon"), "version": m["version"],
            "works": m["counts"]["artworks"], "recognizable": m["counts"]["recognizable"],
            "data_quality": m.get("data_quality", 0), "model_id": m["model_id"],
            "recipe": m.get("recipe"), "built_at": m.get("built_at"),
            "file": z.name, "bytes": z.stat().st_size, "sha256": _sha256(z), "_zip": str(z),
        }
    models = {p["model_id"] for p in best.values()}
    if len(models) > 1:
        raise SystemExit(f"packs construits avec des modèles différents : {models}")
    packs = sorted(best.values(), key=lambda p: (p["country_name"], p["city"], p["name"]))
    cat = {"format": 1, "model_id": models.pop() if models else None,
           "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "packs": [{k: v for k, v in p.items() if not k.startswith("_")} for p in packs]}
    out = root / "catalog.json"
    out.write_text(json.dumps(cat, indent=2, ensure_ascii=False), encoding="utf-8")
    # zips à plat à côté de catalog.json, pour la mise en ligne
    for p in packs:
        src = Path(p["_zip"])
        dst = root / src.name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
    _notice(f"catalog.json : {len(packs)} musées, {sum(p['bytes'] for p in packs) / 1e6:.0f} Mo")
    return out
