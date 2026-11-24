"""Wikidata (CC0) : source universelle pour n'importe quel musée, et pivot pour croiser les données.

Œuvres d'un musée (peintures par défaut ; dessins, aquarelles… au choix) dont la collection (P195) est le
musée ou l'un de ses départements (P361*), ou qui y sont situées (P276).

Pour chaque œuvre : inventaire, image Commons, créateur et ses dates, date, dimensions, et pour des
fiches instructives :
- genre, mouvement, matériaux, sujets représentés, sujet principal, iconographie (Iconclass) ;
- création : lieu, commanditaire, œuvre d'origine ;
- provenance (propriétaires successifs, P127 avec dates) et expositions (P608) ;
- l'artiste : lieux de naissance et de mort, mouvements, maîtres, nationalité ;
- les articles Wikipédia de l'œuvre, de l'artiste, des mouvements et des genres (voir knowledge.py).
Le statut d'exposition reste `unknown` : Wikidata ne dit pas ce qui est accroché.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

from ..records import PDM, Artwork, Museum, Source, normalize_inventory

SPARQL = "https://query.wikidata.org/sparql"
UA = "clara-p0/0.3 (https://github.com/remigl/clara; open data museum app)"
SOURCE = Source(id="wikidata", name="Wikidata", url="https://www.wikidata.org", license="CC0")

# Types d'œuvres (P31)
TYPES = {
    "painting": "Q3305213",
    "drawing": "Q93184",
    "watercolor": "Q18761202",
    "pastel": "Q12043905",
    "miniature": "Q282129",
}
SCOPE_PATHS = {
    "collection": "?item wdt:P195/wdt:P361* wd:{qid} .",
    "location": "?item wdt:P276 wd:{qid} .",
}
BATCH = 250  # détails par lots : une requête unique pour 10 000 tableaux (Louvre) dépasse les limites du service


def scope(qid: str, match: str = "collection", types=("painting",)) -> str:
    values = " ".join(f"wd:{TYPES[t]}" for t in types)
    head = f"VALUES ?type {{ {values} }} ?item wdt:P31 ?type ."
    paths = ["collection", "location"] if match == "both" else [match]
    parts = [f"{{ {head} {SCOPE_PATHS[p].format(qid=qid)} }}" for p in paths]
    return " UNION ".join(parts)


ITEMS_QUERY = "SELECT DISTINCT ?item WHERE {{ {scope} }}"

QUERY = """
SELECT ?item ?itemLabel ?itemDescription ?inv ?image ?creator ?creatorLabel ?birth ?death ?inception ?height ?width
       ?typeLabel WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item wdt:P31 ?type }}
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
SELECT ?item ?v ?vLabel WHERE {{
  VALUES ?item {{ {values} }}
  ?item {path} ?v .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
}}
"""
PROPS = {  # nom interne -> chemin SPARQL (valeurs : [qid, libellé])
    "Genre": "wdt:P136",
    "Mouvement": "wdt:P135",
    "Matériaux": "wdt:P186",
    "Sujets représentés": "wdt:P180",
    "Sujet principal": "wdt:P921",
    "Lieu de création": "wdt:P1071",
    "Commanditaire": "wdt:P88",
    "D'après": "wdt:P144",
    "Inspiré de": "wdt:P941",
    "Expositions": "wdt:P608",
    "_nationality": "wdt:P170/wdt:P27",
}
CREATOR_PROPS = {
    "birth_place": "wdt:P19",
    "death_place": "wdt:P20",
    "movements": "wdt:P135",
    "teachers": "wdt:P1066",
    "students": "wdt:P802",
    "occupations": "wdt:P106",
    "notable_works": "wdt:P800",
}
ICONCLASS_QUERY = """
SELECT ?item ?code WHERE {{ VALUES ?item {{ {values} }} ?item wdt:P1257 ?code . }}
"""
PROVENANCE_QUERY = """
SELECT ?item ?ownerLabel ?start ?end WHERE {{
  VALUES ?item {{ {values} }}
  ?item p:P127 ?st . ?st ps:P127 ?owner .
  OPTIONAL {{ ?st pq:P580 ?start }} OPTIONAL {{ ?st pq:P582 ?end }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
}}
"""
SITELINK_QUERY = """
SELECT ?item ?wiki ?title WHERE {{
  VALUES ?item {{ {values} }}
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


def _sparql(q: str, retries: int = 5) -> list[dict]:
    """Requête SPARQL avec réessais : le service public coupe parfois les réponses ou répond 429/5xx."""
    import requests
    for attempt in range(retries):
        try:
            r = requests.post(SPARQL, data={"query": q, "format": "json"},
                              headers={"User-Agent": UA, "Accept": "application/sparql-results+json"}, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            wait = min(90, 5 * 2 ** attempt)
            print(f"  Wikidata : {e.__class__.__name__}, nouvel essai dans {wait} s", flush=True)
            time.sleep(wait)
    return []


def _v(b, k):
    return b[k]["value"] if k in b else None


def _qid(uri: str | None) -> str | None:
    return uri.rsplit("/", 1)[-1] if uri else None


def _year(s):
    """'1872-01-01T00:00:00Z' -> 1872 ; '-0500-01-01T…' -> -500."""
    m = re.match(r"^\s*([+-]?\d{1,6})-", s or "")
    return int(m.group(1)) if m else None


def _is_label(v: str | None) -> bool:
    return bool(v) and not re.fullmatch(r"Q\d+", v) and not v.startswith("http")


LABEL_QUERY = """
SELECT DISTINCT ?m WHERE {{
  VALUES ?label {{ "{name}"@{lang} "{name}"@en "{name}"@de "{name}"@fr }}
  ?m rdfs:label|skos:altLabel ?label .
  ?m wdt:P31/wdt:P279* wd:Q33506 .
}} LIMIT 3
"""


MUSEUM_WORDS = ("museum", "musée", "museo", "gallery", "galerie", "galleria", "galería", "pinacot",
                "kunsthalle", "kunsthaus", "collection", "kunstmuseum", "museu")


def resolve_qid(search: str, lang: str = "de", retries: int = 4) -> str:
    """Identifiant Wikidata d'un musée d'après son nom : API de recherche, puis requête SPARQL sur les libellés
    (l'API de recherche limite fortement les robots quand plusieurs constructions tournent)."""
    import requests
    for attempt in range(retries):
        try:
            r = requests.get("https://www.wikidata.org/w/api.php",
                             params={"action": "wbsearchentities", "search": search, "language": lang, "type": "item",
                                     "limit": 7, "format": "json"}, headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 200 and "error" not in r.json():
                hits = r.json().get("search") or []
                # préférer un résultat décrit comme un musée (évite l'homonyme : bâtiment, gare, personne…)
                museums = [h for h in hits if any(k in (h.get("description") or "").lower() for k in MUSEUM_WORDS)]
                hits = museums or hits
                if hits:
                    print(f"  Wikidata : « {search} » -> {hits[0]['id']} ({hits[0].get('label')}, "
                          f"{hits[0].get('description')})")
                    return hits[0]["id"]
                break
        except (requests.RequestException, ValueError):
            pass
        time.sleep(min(60, 5 * 2 ** attempt))
    name = search.replace('"', '\\"')
    hits = [_qid(_v(b, "m")) for b in _sparql(LABEL_QUERY.format(name=name, lang=lang))]
    if not hits:
        raise ValueError(f"aucun musée Wikidata nommé « {search} »")
    print(f"  Wikidata (SPARQL) : « {search} » -> {hits[0]}" + (f" (autres : {hits[1:]})" if len(hits) > 1 else ""))
    return hits[0]


def museum(qid: str, museum_id: str, lang: str = "fr", city: str | None = None,
           country_name: str | None = None, country: str | None = None) -> Museum:
    b = _sparql(MUSEUM_QUERY.format(qid=qid, lang=lang))
    if not b:
        raise ValueError(f"musée {qid} introuvable sur Wikidata")
    b = b[0]
    lat = lon = 0.0
    if _v(b, "coord"):  # "Point(lon lat)"
        lon, lat = map(float, _v(b, "coord")[6:-1].split())
    return Museum(id=museum_id, name=_v(b, "label"), city=city or _v(b, "cityLabel") or "",
                  country=country or _v(b, "countryCode") or "", country_name=country_name,
                  lat=lat, lon=lon, website=_v(b, "website"), wikidata_qid=qid, data_quality=0)


def _batches(qids: list[str]):
    for i in range(0, len(qids), BATCH):
        yield i, " ".join(f"wd:{q}" for q in qids[i:i + BATCH])


def _prop_values(values: str, path: str, lang: str) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    for b in _sparql(PROP_QUERY.format(values=values, lang=lang, path=path)):
        q, vq, v = _qid(_v(b, "item")), _qid(_v(b, "v")), _v(b, "vLabel")
        if q and _is_label(v) and [vq, v] not in out.setdefault(q, []):
            out[q].append([vq, v])
    return out


def _sitelinks(values: str, lang: str, into: dict) -> None:
    for b in _sparql(SITELINK_QUERY.format(values=values, lang=lang)):
        wiki = "en" if "en.wikipedia" in (_v(b, "wiki") or "") else lang
        into.setdefault(_qid(_v(b, "item")), {})[wiki] = _v(b, "title")


def _safe(what: str, fn):
    """Une donnée annexe en échec ne bloque pas le pack."""
    try:
        return fn()
    except Exception as e:
        print(f"  Wikidata : {what} ignoré pour ce lot ({e.__class__.__name__})", flush=True)
        return None


def fetch(qid: str, out_dir: str | Path, lang: str = "fr", match: str = "collection", types=("painting",)) -> dict:
    items = sorted({_qid(_v(b, "item")) for b in _sparql(ITEMS_QUERY.format(scope=scope(qid, match, types)))})
    print(f"  Wikidata : {len(items)} œuvres ({match}, {', '.join(types)}), détails par lots de {BATCH}", flush=True)
    rows: list[dict] = []
    props: dict[str, dict] = {name: {} for name in PROPS}
    iconclass: dict[str, list[str]] = {}
    provenance: dict[str, list[list]] = {}
    links: dict[str, dict[str, str]] = {}
    for i, values in _batches(items):
        rows += _sparql(QUERY.format(values=values, lang=lang))
        for name, path in PROPS.items():
            got = _safe(name, lambda: _prop_values(values, path, lang)) or {}
            for q, vals in got.items():
                props[name].setdefault(q, []).extend(v for v in vals if v not in props[name].get(q, []))
        for b in _safe("iconclass", lambda: _sparql(ICONCLASS_QUERY.format(values=values))) or []:
            c = _v(b, "code")
            if c and c not in iconclass.setdefault(_qid(_v(b, "item")), []):
                iconclass[_qid(_v(b, "item"))].append(c)
        for b in _safe("provenance", lambda: _sparql(PROVENANCE_QUERY.format(values=values, lang=lang))) or []:
            owner = _v(b, "ownerLabel")
            if _is_label(owner):
                e = [owner, _year(_v(b, "start")), _year(_v(b, "end"))]
                if e not in provenance.setdefault(_qid(_v(b, "item")), []):
                    provenance[_qid(_v(b, "item"))].append(e)
        _safe("articles Wikipédia", lambda: _sitelinks(values, lang, links))
        print(f"  Wikidata : œuvres {min(i + BATCH, len(items))}/{len(items)}", flush=True)
        time.sleep(0.3)

    # artistes
    creators_q = sorted({_qid(_v(b, "creator")) for b in rows if _v(b, "creator")} - {None})
    creators: dict[str, dict] = {c: {"sitelinks": {}} for c in creators_q}
    for i, values in _batches(creators_q):
        for key, path in CREATOR_PROPS.items():
            for q, vals in (_safe(key, lambda: _prop_values(values, path, lang)) or {}).items():
                creators[q][key] = vals
        tmp: dict = {}
        _safe("articles des artistes", lambda: _sitelinks(values, lang, tmp))
        for q, t in tmp.items():
            creators.setdefault(q, {"sitelinks": {}})["sitelinks"] = t
        print(f"  Wikidata : artistes {min(i + BATCH, len(creators_q))}/{len(creators_q)}", flush=True)
        time.sleep(0.3)

    # notions (mouvements, genres) : pour le glossaire « pour comprendre »
    topics: dict[str, dict] = {}
    for name in ("Mouvement", "Genre"):
        for vals in props[name].values():
            for vq, label in vals:
                topics.setdefault(vq, {"label": label, "sitelinks": {}})
    for c in creators.values():
        for vq, label in c.get("movements", []):
            topics.setdefault(vq, {"label": label, "sitelinks": {}})
    topic_q = sorted(t for t in topics if t)
    for i, values in _batches(topic_q):
        tmp = {}
        _safe("articles des notions", lambda: _sitelinks(values, lang, tmp))
        for q, t in tmp.items():
            topics[q]["sitelinks"] = t

    raw = {"format": 2, "lang": lang, "main": rows, "props": props, "iconclass": iconclass,
           "provenance": provenance, "sitelinks": links, "creators": creators, "topics": topics}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return raw


def load_raw(out_dir: str | Path) -> dict:
    raw = json.loads((Path(out_dir) / "raw.json").read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {"main": raw, "props": {}, "sitelinks": {}, "lang": "fr"}


def _labels(vals) -> list[str]:
    """Valeurs d'une propriété : [[qid, libellé], …] (format 2) ou [libellé, …] (format 1)."""
    return [v[1] if isinstance(v, list) else v for v in vals or []]


def _qids(vals) -> list[str]:
    return [v[0] for v in vals or [] if isinstance(v, list) and v[0]]


def _span(a, b) -> str:
    if a and b:
        return f"{a}–{b}"
    if a:
        return f"depuis {a}"
    if b:
        return f"jusqu'en {b}"
    return ""


def select(raw, museum_id: str, pd_years: int = 70, summaries: dict | None = None,
           know: dict | None = None) -> list[Artwork]:
    """Une œuvre par item, avec les détails des fiches.

    know (knowledge.gather) : textes Wikipédia de l'œuvre, de l'artiste et des notions, libellés Iconclass.
    summaries : ancien format {qid: (texte, langue)}, encore accepté."""
    if isinstance(raw, list):
        raw = {"main": raw, "props": {}, "sitelinks": {}}
    know = know or {}
    art_txt = know.get("art") or {q: t for q, t in (summaries or {}).items()}
    creator_txt = know.get("creator", {})
    topic_txt = know.get("topic", {})
    icon_lbl = know.get("iconclass", {})
    rows, props = raw["main"], raw.get("props", {})
    creators, topics = raw.get("creators", {}), raw.get("topics", {})
    by_item: dict[str, Artwork] = {}
    this_year = date.today().year
    for b in rows:
        qid = _qid(_v(b, "item"))
        if not qid or qid in by_item:
            continue
        death = _year(_v(b, "death"))
        img = _v(b, "image")
        if img:
            img = img.replace("http://", "https://") + "?width=1024"
        # Domaine public si l'artiste est mort depuis plus de 70 ans (au 1er janvier) ; sinon droits inconnus
        rights = PDM if death is not None and death + pd_years < this_year else None
        label = _v(b, "itemLabel")
        type_label = _v(b, "typeLabel")
        creator_q = _qid(_v(b, "creator"))
        a = Artwork(
            museum_id=museum_id,
            inventory_number=_v(b, "inv") or qid,
            title=label if _is_label(label) else "Sans titre",
            source_id=SOURCE.id,
            artist_display=_v(b, "creatorLabel") if _is_label(_v(b, "creatorLabel")) else "Anonyme",
            artist_qid=creator_q,
            artist_birth_year=_year(_v(b, "birth")),
            artist_death_year=death,
            date_start=_year(_v(b, "inception")),
            date_display=str(_year(_v(b, "inception"))) if _v(b, "inception") else None,
            height_cm=float(_v(b, "height")) if _v(b, "height") else None,
            width_cm=float(_v(b, "width")) if _v(b, "width") else None,
            object_type="painting" if type_label in (None, "peinture", "painting", "tableau") else type_label,
            wikidata_qid=qid,
            source_url=f"https://www.wikidata.org/wiki/{qid}",
            image_url=img,
            image_rights_uri=rights,
            credit_line="Image : Wikimedia Commons" + (" (domaine public)" if rights else ""),
        )
        by_item[qid] = a
        p = {name: props.get(name, {}).get(qid) for name in PROPS}

        # détails courts
        facts = []
        for name in ("Genre", "Mouvement", "Matériaux", "Sujets représentés"):
            vals = _labels(p[name])
            if vals:
                facts.append((name, ", ".join(vals[:8])))
        a.facts = facts
        if p["Matériaux"]:
            a.medium = ", ".join(_labels(p["Matériaux"])[:4])
        if p["_nationality"]:
            a.artist_nationality = _labels(p["_nationality"])[0]

        # sections thématiques
        sections = []
        icon = [icon_lbl.get(c) or c for c in raw.get("iconclass", {}).get(qid, [])]
        iconography = [("Sujet", v) for v in _labels(p["Sujet principal"])]
        iconography += [("Iconclass", v) for v in icon if v]
        if iconography:
            sections.append(("Iconographie", iconography))
        creation = [(name, ", ".join(_labels(p[name]))) for name in ("Lieu de création", "Commanditaire", "D'après",
                                                                     "Inspiré de") if p[name]]
        if creation:
            sections.append(("Création", creation))
        prov = sorted(raw.get("provenance", {}).get(qid, []), key=lambda e: (e[1] or 9999, e[2] or 9999))
        if prov:
            sections.append(("Provenance", [(_span(e[1], e[2]) or "—", e[0]) for e in prov]))
        if p["Expositions"]:
            sections.append(("Expositions", [("", v) for v in _labels(p["Expositions"])[:12]]))
        a.sections = sections

        # articles complets : description, contexte, à observer, anecdotes (en tête des sections)
        ath = know.get("art_themes", {}).get(qid)
        if ath:
            credit = ("Source", f"Texte : Wikipédia ({ath[1]}), licence CC BY-SA 4.0")
            order = ["Description", "À observer", "Contexte et histoire", "Anecdotes"]
            a.sections = [(k, [("", ath[0][k]), credit]) for k in order if ath[0].get(k)] + a.sections
        depicts = _labels(p["Sujets représentés"])
        if depicts and not (ath and ath[0].get("À observer")):
            a.sections.insert(0, ("À observer", [("", "Dans ce tableau, on peut repérer : " + ", ".join(depicts[:12]) + ".")]))

        # texte de présentation
        t = art_txt.get(qid)
        if t:
            a.description, lang = t[0], t[1]
            a.description_credit = f"Texte : Wikipédia ({lang}), licence CC BY-SA 4.0"
        elif _v(b, "itemDescription"):
            d = _v(b, "itemDescription")
            a.description = d[:1].upper() + d[1:] + "."
            a.description_credit = "Texte : Wikidata (CC0)"

        # l'artiste
        c = creators.get(creator_q) or {}
        if c:
            a.artist_birth_place = (_labels(c.get("birth_place")) or [None])[0]
            a.artist_death_place = (_labels(c.get("death_place")) or [None])[0]
            a.artist_movements = ", ".join(_labels(c.get("movements"))[:5]) or None
            a.artist_teachers = ", ".join(_labels(c.get("teachers"))[:6]) or None
            a.artist_students = ", ".join(_labels(c.get("students"))[:6]) or None
            a.artist_occupations = ", ".join(_labels(c.get("occupations"))[:4]) or None
            a.artist_notable_works = ", ".join(_labels(c.get("notable_works"))[:8]) or None
        cth = know.get("creator_themes", {}).get(creator_q)
        if cth:
            credit = f"Texte : Wikipédia ({cth[1]}), licence CC BY-SA 4.0"
            a.artist_style = cth[0].get("Style")
            a.artist_anecdotes = cth[0].get("Anecdotes")
            a.artist_extra_credit = credit
        ct = creator_txt.get(creator_q)
        if ct:
            a.artist_bio = ct[0]
            a.artist_bio_credit = f"Texte : Wikipédia ({ct[1]}), licence CC BY-SA 4.0"

        # pour comprendre : mouvements et genres expliqués
        glossary = []
        topic_ids = _qids(p["Mouvement"]) + _qids(p["Genre"]) + _qids(c.get("movements"))
        for tq in dict.fromkeys(topic_ids):
            tt = topic_txt.get(tq)
            if tt and topics.get(tq):
                glossary.append((topics[tq]["label"], tt[0], f"Texte : Wikipédia ({tt[1]}), licence CC BY-SA 4.0"))
        a.glossary = glossary[:6]
    return list(by_item.values())


def enrich_by_inventory(artworks: list[Artwork], raw: dict, know: dict, museum_id: str) -> int:
    """Croise des œuvres d'une autre source (ex. API SMK) avec Wikidata par numéro d'inventaire :
    ajoute l'artiste, le contexte, l'iconographie, la provenance et le texte Wikipédia. Retourne le nombre
    d'œuvres enrichies. Le texte du musée, s'il existe, reste le texte principal."""
    wd = {normalize_inventory(a.inventory_number): a for a in select(raw, museum_id, know=know)}
    n = 0
    for a in artworks:
        w = wd.get(normalize_inventory(a.inventory_number))
        if not w:
            continue
        n += 1
        a.wikidata_qid = w.wikidata_qid
        for f in ("artist_qid", "artist_bio", "artist_bio_credit", "artist_birth_place", "artist_death_place",
                  "artist_movements", "artist_teachers", "artist_students", "artist_occupations",
                  "artist_notable_works", "artist_style", "artist_anecdotes", "artist_extra_credit"):
            if getattr(a, f, None) in (None, "") and getattr(w, f, None):
                setattr(a, f, getattr(w, f))
        have = {k for k, _ in a.facts}
        a.facts = a.facts + [f for f in w.facts if f[0] not in have]
        a.sections = a.sections + w.sections
        a.glossary = a.glossary + [g for g in w.glossary if g[0] not in {x[0] for x in a.glossary}]
        if w.description and w.description_credit and "Wikipédia" in w.description_credit:
            if a.description:
                a.sections.append(("Sur Wikipédia", [("", w.description), ("Source", w.description_credit)]))
            else:
                a.description, a.description_credit = w.description, w.description_credit
    return n
