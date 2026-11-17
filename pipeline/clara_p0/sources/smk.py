"""SMK — Statens Museum for Kunst (Copenhague). API ouverte : https://api.smk.dk

- Données et images des œuvres du domaine public : libres (Public Domain Mark).
- `current_location_name` (ex. « Sal 205 ») indique la salle ; s'il manque, l'œuvre n'est pas exposée.
  -> statut d'exposition au niveau 3 (salle) dès le premier build.

Les noms de champs utilisés ici viennent de la documentation SMK ; ils sont lus de façon défensive
et la réponse brute est sauvegardée (`raw.jsonl`) pour corriger facilement un champ manquant.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from ..records import PDM, Artwork, Museum, Source

API = "https://api.smk.dk/api/v1/art/search/"
UA = "clara-p0/0.1 (https://github.com/remigl/clara)"

MUSEUM = Museum(
    id="dk-smk", name="National Gallery of Denmark", name_local="Statens Museum for Kunst",
    city="Copenhague", country="DK", lat=55.68853, lon=12.57836, website="https://www.smk.dk",
    data_quality=3,
)
SOURCE = Source(id="smk-api", name="SMK Open API", url="https://api.smk.dk", license="CC0 / PDM")

PAINTING_WORDS = ("maleri", "painting", "oil painting", "oliemaleri")


def _first(lst, key=None):
    if isinstance(lst, list) and lst:
        return lst[0].get(key) if key and isinstance(lst[0], dict) else lst[0]
    return None


def _year(s):
    if not s:
        return None
    m = re.match(r"-?\d{3,4}", str(s))
    return int(m.group()) if m else None


def _title(item):
    titles = item.get("titles") or []
    for lang in ("en", "da"):
        for t in titles:
            if t.get("language") == lang and t.get("title"):
                return t["title"]
    return _first(titles, "title") or item.get("object_number")


def _dimensions(item):
    h = w = None
    for d in item.get("dimensions") or []:
        typ, val, unit = (d.get("type") or "").lower(), d.get("value"), (d.get("unit") or "cm").lower()
        try:
            v = float(str(val).replace(",", "."))
        except (TypeError, ValueError):
            continue
        v = v / 10 if unit == "mm" else v
        if typ in ("højde", "height") and h is None:
            h = v
        elif typ in ("bredde", "width") and w is None:
            w = v
    return h, w


def _image_url(item, size: int):
    iiif = item.get("image_iiif_id")
    if iiif:
        return f"{iiif.rstrip('/')}/full/!{size},{size}/0/default.jpg"
    return item.get("image_thumbnail") or item.get("image_native")


def is_painting(item) -> bool:
    names = " ".join((n.get("name") or "").lower() for n in item.get("object_names") or [] if isinstance(n, dict))
    return any(w in names for w in PAINTING_WORDS)


def to_artwork(item, image_size: int = 1024) -> Artwork:
    prod = _first(item.get("production") or []) or {}
    pdate = _first(item.get("production_date") or []) or {}
    h, w = _dimensions(item)
    room = item.get("current_location_name")
    pd = bool(item.get("public_domain"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Artwork(
        museum_id=MUSEUM.id,
        inventory_number=item["object_number"],
        title=_title(item),
        source_id=SOURCE.id,
        artist_display=prod.get("creator") or "Anonyme",
        artist_birth_year=_year(prod.get("creator_date_of_birth")),
        artist_death_year=_year(prod.get("creator_date_of_death")),
        date_display=pdate.get("period"),
        date_start=_year(pdate.get("start")),
        date_end=_year(pdate.get("end")),
        medium=", ".join(t for t in (item.get("techniques") or []) if isinstance(t, str)) or None,
        height_cm=h, width_cm=w,
        object_type="painting" if is_painting(item) else (_first(item.get("object_names") or [], "name") or "other"),
        collection=_first(item.get("responsible_department") or []) if isinstance(item.get("responsible_department"), list)
        else item.get("responsible_department"),
        room=room,
        exhibition_status="on_display" if room else "not_on_display",
        status_checked_at=now,
        source_url=f"https://open.smk.dk/artwork/image/{item['object_number']}",
        image_url=_image_url(item, image_size) if item.get("has_image", True) else None,
        image_rights_uri=PDM if pd else item.get("rights"),
        credit_line=f"{prod.get('creator') or 'Anonyme'}, SMK – Statens Museum for Kunst" + (" (domaine public)" if pd else ""),
    )


def fetch(out_dir: str | Path, filters=("[has_image:true]", "[public_domain:true]"), rows: int = 100,
          max_items: int | None = None, pause: float = 0.2) -> list[dict]:
    """Télécharge les métadonnées brutes (pagination) et les sauvegarde dans raw.jsonl."""
    import requests

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.jsonl"
    items, offset, found = [], 0, None
    s = requests.Session()
    s.headers["User-Agent"] = UA
    while found is None or offset < found:
        params = {"keys": "*", "offset": offset, "rows": rows, "lang": "en", "filters": ",".join(filters)}
        r = s.get(API, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        found = data.get("found", 0)
        batch = data.get("items") or []
        if not batch:
            break
        items.extend(batch)
        offset += len(batch)
        print(f"  SMK : {offset}/{found}", flush=True)
        if max_items and len(items) >= max_items:
            items = items[:max_items]
            break
        time.sleep(pause)
    with open(raw_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items


def load_raw(out_dir: str | Path) -> list[dict]:
    with open(Path(out_dir) / "raw.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select(items: list[dict], paintings_only: bool = True, on_display_only: bool = True) -> list[Artwork]:
    arts = []
    for it in items:
        if not it.get("object_number"):
            continue
        if paintings_only and not is_painting(it):
            continue
        if on_display_only and not it.get("current_location_name"):
            continue
        arts.append(to_artwork(it))
    return arts
