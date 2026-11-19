"""Wikidata (CC0) : source universelle de niveau 0 pour n'importe quel musée.

Peintures (Q3305213) dont la collection (P195) est le musée ou l'un de ses départements (P361*,
ex. « Département des Peintures du Louvre »), avec n° d'inventaire (P217),
image Commons (P18), créateur (P170) et ses dates, date (P571), dimensions, plus les détails
des fiches : genre (P136), mouvement (P135), matériaux (P186), sujets (P180), nationalité de
l'artiste (P27), et le résumé Wikipédia (CC BY-SA) quand l'œuvre a un article.
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
SELECT ?item ?itemLabel ?itemDescription ?inv ?image ?creator ?creatorLabel ?birth ?death ?inception ?height ?width WHERE {{
  ?item wdt:P31 wd:Q3305213 ; wdt:P195/wdt:P361* wd:{qid} .
  OPTIONAL {{ ?item wdt:P217 ?inv }}
  OPTIONAL {{ ?item wdt:P18 ?image }}
  OPTIONAL {{ ?item wdt:P170 ?creator . OPTIONAL {{ ?creator wdt:P569 ?birth }} OPTIONAL {{ ?creator wdt:P570 ?death }} }}
  OPTIONAL {{ ?item wdt:P571 ?inception }}
  OPTIONAL {{ ?item wdt:P2048 ?height }}
  OPTIONAL {{ ?item wdt:P2049 ?width }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
}}
"""

# une requête légère par propriété à valeurs multiples (évite l'explosion combinatoire des OPTIONAL)
PROP_QUERY = """
SELECT ?item ?vLabel WHERE {{
  ?item wdt:P31 wd:Q3305213 ; wdt:P195/wdt:P361* wd:{qid} ; {path} ?v .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
}}
"""
PROPS = {  # libellé dans la fiche -> chemin SPARQL
    "Genre": "wdt:P136",
    "Mouvement": "wdt:P135",
    "Matériaux": "wdt:P186",
    "Sujets représentés": "wdt:P180",
    "_nationality": "wdt:P170/wdt:P27",
}
SITELINK_QUERY = """
SELECT ?item ?wiki ?title WHERE {{
  ?item wdt:P31 wd:Q3305213 ; wdt:P195/wdt:P361* wd:{qid} .
  ?article schema:about ?item ; schema:isPartOf ?wiki ; schema:name ?title .
  VALUES ?wiki {{ <https://{lang}.wikipedia.org/> <https://en.wikipedia.org/> }}
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


def museum(qid: str, museum_id: str, lang: str = "fr", city: str | None = None,
           country_name: str | None = None) -> Museum:
    b = _sparql(MUSEUM_QUERY.format(qid=qid, lang=lang))
    if not b:
        raise ValueError(f"musée {qid} introuvable sur Wikidata")
    b = b[0]
    lat = lon = 0.0
    if _v(b, "coord"):  # "Point(lon lat)"
        lon, lat = map(float, _v(b, "coord")[6:-1].split())
    return Museum(id=museum_id, name=_v(b, "label"), city=city or _v(b, "cityLabel") or "",
                  country=_v(b, "countryCode") or "", country_name=country_name,
                  lat=lat, lon=lon, website=_v(b, "website"), wikidata_qid=qid, data_quality=0)


def fetch(qid: str, out_dir: str | Path, lang: str = "fr") -> dict:
    rows = _sparql(QUERY.format(qid=qid, lang=lang))
    props: dict[str, dict[str, list[str]]] = {}
    for name, path in PROPS.items():
        try:
            res = _sparql(PROP_QUERY.format(qid=qid, lang=lang, path=path))
        except Exception as e:  # une propriété en échec ne bloque pas le pack
            print(f"  Wikidata : {name} ignoré ({e.__class__.__name__})")
            continue
        d: dict[str, list[str]] = {}
        for b in res:
            q, v = _qid(_v(b, "item")), _v(b, "vLabel")
            if q and v and not re.fullmatch(r"Q\d+", v) and v not in d.setdefault(q, []):
                d[q].append(v)
        props[name] = d
    links: dict[str, dict[str, str]] = {}
    try:
        for b in _sparql(SITELINK_QUERY.format(qid=qid, lang=lang)):
            wiki = "en" if "en.wikipedia" in (_v(b, "wiki") or "") else lang
            links.setdefault(_qid(_v(b, "item")), {})[wiki] = _v(b, "title")
    except Exception as e:
        print(f"  Wikidata : articles Wikipédia ignorés ({e.__class__.__name__})")
    raw = {"main": rows, "props": props, "sitelinks": links, "lang": lang}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return raw


def load_raw(out_dir: str | Path) -> dict:
    raw = json.loads((Path(out_dir) / "raw.json").read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {"main": raw, "props": {}, "sitelinks": {}, "lang": "fr"}


def _qid(uri: str | None) -> str | None:
    return uri.rsplit("/", 1)[-1] if uri else None


def wikipedia_summaries(raw: dict, cache_file: str | Path, pause: float = 0.05) -> dict[str, tuple[str, str]]:
    """Résumé (introduction) de l'article Wikipédia de chaque œuvre qui en a un, langue du pack d'abord.
    Retourne {qid: (texte, langue)}. Mis en cache : les articles ne sont demandés qu'une fois."""
    import time

    import requests

    cache_file = Path(cache_file)
    cache: dict[str, list] = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    lang = raw.get("lang", "fr")
    s = requests.Session()
    s.headers["User-Agent"] = UA
    todo = [(q, l) for q, l in raw.get("sitelinks", {}).items() if q not in cache]
    if todo:
        print(f"  Wikipédia : {len(todo)} résumés à télécharger", flush=True)
    for n, (q, titles) in enumerate(todo, 1):
        for wl in (lang, "en"):
            title = titles.get(wl)
            if not title:
                continue
            try:
                r = s.get(f"https://{wl}.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
                          timeout=30)
                if r.status_code == 200:
                    ext = (r.json().get("extract") or "").strip()
                    if len(ext) > 60:
                        cache[q] = [ext, wl]
                        break
            except (requests.RequestException, ValueError):
                pass
        else:
            cache.setdefault(q, ["", ""])
        if n % 100 == 0:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"  Wikipédia : {n}/{len(todo)}", flush=True)
        time.sleep(pause)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {q: (t, l) for q, (t, l) in cache.items() if t}


def select(raw, museum_id: str, pd_years: int = 70, summaries: dict | None = None) -> list[Artwork]:
    """Une œuvre par item (dédoublonnage des lignes multi-créateurs), avec les détails des fiches."""
    if isinstance(raw, list):
        raw = {"main": raw, "props": {}, "sitelinks": {}}
    rows, props, summaries = raw["main"], raw.get("props", {}), summaries or {}
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
            credit_line="Image : Wikimedia Commons" + (" (domaine public)" if rights else ""),
        )
        a = by_item[qid]
        facts = []
        for name in ("Genre", "Mouvement", "Matériaux", "Sujets représentés"):
            vals = props.get(name, {}).get(qid) or []
            if vals:
                facts.append((name, ", ".join(vals[:6])))
        a.facts = facts
        mats = props.get("Matériaux", {}).get(qid)
        if mats:
            a.medium = ", ".join(mats[:4])
        nat = props.get("_nationality", {}).get(qid)
        if nat:
            a.artist_nationality = nat[0]
        summ = summaries.get(qid)
        if summ:
            a.description = summ[0]
            a.description_credit = f"Texte : Wikipédia ({summ[1]}), licence CC BY-SA 4.0"
        elif _v(b, "itemDescription"):
            a.description = _v(b, "itemDescription")[:1].upper() + _v(b, "itemDescription")[1:] + "."
            a.description_credit = "Texte : Wikidata (CC0)"
    return list(by_item.values())
