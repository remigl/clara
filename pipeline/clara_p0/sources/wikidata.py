"""Wikidata (CC0) : source universelle de niveau 0 pour n'importe quel musée.

Peintures (Q3305213) dont la collection (P195) est le musée, avec n° d'inventaire (P217),
image Commons (P18), créateur (P170) et sa date de mort (P570), date (P571), dimensions.
Le statut d'exposition reste `unknown` : Wikidata ne dit pas ce qui est accroché.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from ..records import PDM, Artwork, Museum, Source

SPARQL = "https://query.wikidata.org/sparql"
UA = "clara-p0/0.1 (https://github.com/remigl/clara)"
SOURCE = Source(id="wikidata", name="Wikidata", url="https://www.wikidata.org", license="CC0")

QUERY = """
SELECT ?item ?itemLabel ?inv ?image ?creator ?creatorLabel ?birth ?death ?inception ?height ?width WHERE {{
  ?item wdt:P31 wd:Q3305213 ; wdt:P195 wd:{qid} .
  OPTIONAL {{ ?item p:P217 ?invSt . ?invSt ps:P217 ?inv . OPTIONAL {{ ?invSt pq:P195 ?invColl . }}
             FILTER(!BOUND(?invColl) || ?invColl = wd:{qid}) }}
  OPTIONAL {{ ?item wdt:P18 ?image }}
  OPTIONAL {{ ?item wdt:P170 ?creator . OPTIONAL {{ ?creator wdt:P569 ?birth }} OPTIONAL {{ ?creator wdt:P570 ?death }} }}
  OPTIONAL {{ ?item wdt:P571 ?inception }}
  OPTIONAL {{ ?item wdt:P2048 ?height }}
  OPTIONAL {{ ?item wdt:P2049 ?width }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
}}
"""

MUSEUM_QUERY = """
SELECT ?label ?coord ?website ?cityLabel ?countryCode WHERE {{
  wd:{qid} rdfs:label ?label . FILTER(LANG(?label) = "{lang}")
  OPTIONAL {{ wd:{qid} wdt:P625 ?coord }}
  OPTIONAL {{ wd:{qid} wdt:P856 ?website }}
  OPTIONAL {{ wd:{qid} wdt:P131 ?city }}
  OPTIONAL {{ wd:{qid} wdt:P17 ?country . ?country wdt:P297 ?countryCode }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
}} LIMIT 1
"""


def _sparql(q: str) -> list[dict]:
    import requests
    r = requests.get(SPARQL, params={"query": q, "format": "json"},
                     headers={"User-Agent": UA, "Accept": "application/sparql-results+json"}, timeout=180)
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def _v(b, k):
    return b[k]["value"] if k in b else None


def _year(s):
    """'1872-01-01T00:00:00Z' -> 1872 ; '-0500-01-01T…' -> -500."""
    m = re.match(r"^\s*([+-]?\d{1,6})-", s or "")
    return int(m.group(1)) if m else None


def museum(qid: str, museum_id: str, lang: str = "fr") -> Museum:
    b = _sparql(MUSEUM_QUERY.format(qid=qid, lang=lang))
    if not b:
        raise ValueError(f"musée {qid} introuvable sur Wikidata")
    b = b[0]
    lat = lon = 0.0
    if _v(b, "coord"):  # "Point(lon lat)"
        lon, lat = map(float, _v(b, "coord")[6:-1].split())
    return Museum(id=museum_id, name=_v(b, "label"), city=_v(b, "cityLabel") or "", country=_v(b, "countryCode") or "",
                  lat=lat, lon=lon, website=_v(b, "website"), wikidata_qid=qid, data_quality=0)


def fetch(qid: str, out_dir: str | Path, lang: str = "fr") -> list[dict]:
    rows = _sparql(QUERY.format(qid=qid, lang=lang))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def load_raw(out_dir: str | Path) -> list[dict]:
    return json.loads((Path(out_dir) / "raw.json").read_text(encoding="utf-8"))


def select(rows: list[dict], museum_id: str, pd_years: int = 70) -> list[Artwork]:
    """Une œuvre par item (dédoublonnage des lignes multi-créateurs)."""
    by_item: dict[str, Artwork] = {}
    this_year = date.today().year
    for b in rows:
        item = _v(b, "item")
        if not item:
            continue
        qid = item.rsplit("/", 1)[-1]
        if qid in by_item:
            continue
        death = _year(_v(b, "death"))
        img = _v(b, "image")
        if img:
            img = img.replace("http://", "https://") + "?width=1024"
        # Domaine public si l'artiste est mort depuis plus de 70 ans (au 1er janvier) ; sinon droits inconnus
        rights = PDM if death is not None and death + pd_years < this_year else None
        label = _v(b, "itemLabel")
        by_item[qid] = Artwork(
            museum_id=museum_id,
            inventory_number=_v(b, "inv") or qid,
            title=label if label and label != qid else "Sans titre",
            source_id=SOURCE.id,
            artist_display=_v(b, "creatorLabel") or "Anonyme",
            artist_qid=(_v(b, "creator") or "").rsplit("/", 1)[-1] or None,
            artist_birth_year=_year(_v(b, "birth")),
            artist_death_year=death,
            date_start=_year(_v(b, "inception")),
            date_display=str(_year(_v(b, "inception"))) if _v(b, "inception") else None,
            height_cm=float(_v(b, "height")) if _v(b, "height") else None,
            width_cm=float(_v(b, "width")) if _v(b, "width") else None,
            wikidata_qid=qid,
            source_url=f"https://www.wikidata.org/wiki/{qid}",
            image_url=img,
            image_rights_uri=rights,
            credit_line=f"{_v(b, 'creatorLabel') or 'Anonyme'} — image via Wikimedia Commons",
        )
    return list(by_item.values())
