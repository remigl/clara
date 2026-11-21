"""Textes de référence croisés avec Wikidata : introductions Wikipédia (œuvre, artiste, mouvements, genres)
et libellés Iconclass (classification iconographique de référence, https://iconclass.org).

Tout est mis en cache sur disque : une page n'est demandée qu'une fois, même d'un pack à l'autre.
Licences : Wikipédia CC BY-SA 4.0 (crédit affiché dans l'app), Iconclass ODC-By.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote

UA = "clara-p0/0.1 (https://github.com/remigl/clara)"


def _trim(text: str, max_chars: int) -> str:
    """Paragraphes entiers jusqu'à max_chars (au moins le premier)."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, n = [], 0
    for p in paras:
        if out and n + len(p) > max_chars:
            break
        out.append(p)
        n += len(p)
    return "\n\n".join(out)


class Cache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


def wiki_intros(titles: dict[str, list[str]], cache: Cache, pause: float = 0.1) -> None:
    """Introductions d'articles, par lots de 20 (API extracts). titles : {langue: [titres]}.
    Résultat dans cache.data['wiki'][f'{langue}:{titre}'] (chaîne vide si pas de texte)."""
    import requests

    wiki = cache.data.setdefault("wiki", {})
    s = requests.Session()
    s.headers["User-Agent"] = UA
    for lang, ts in titles.items():
        todo = sorted({t for t in ts if f"{lang}:{t}" not in wiki})
        if todo:
            print(f"  Wikipédia ({lang}) : {len(todo)} articles", flush=True)
        for i in range(0, len(todo), 20):
            chunk = todo[i:i + 20]
            for attempt in range(4):
                try:
                    r = s.get(f"https://{lang}.wikipedia.org/w/api.php", timeout=60, params={
                        "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1,
                        "format": "json", "formatversion": 2, "titles": "|".join(chunk)})
                    if r.status_code in (429, 503):
                        time.sleep(10 * (attempt + 1))
                        continue
                    q = r.json().get("query", {})
                    break
                except Exception:
                    time.sleep(5 * (attempt + 1))
            else:
                continue
            # titre demandé -> titre final (normalisation, redirections)
            alias = {t: t for t in chunk}
            for m in q.get("normalized", []) + q.get("redirects", []):
                for k, v in list(alias.items()):
                    if v == m["from"]:
                        alias[k] = m["to"]
            pages = {p["title"]: (p.get("extract") or "") for p in q.get("pages", [])}
            for t in chunk:
                wiki[f"{lang}:{t}"] = pages.get(alias[t], "")
            if (i // 20) % 10 == 9:
                cache.save()
                print(f"  Wikipédia ({lang}) : {min(i + 20, len(todo))}/{len(todo)}", flush=True)
            time.sleep(pause)
    cache.save()


def iconclass_labels(codes: set[str], cache: Cache, lang: str = "fr", pause: float = 0.1) -> dict[str, str]:
    import requests

    store = cache.data.setdefault("iconclass", {})
    todo = sorted(c for c in codes if c not in store)
    if todo:
        print(f"  Iconclass : {len(todo)} codes", flush=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    for n, code in enumerate(todo, 1):
        try:
            r = s.get(f"https://iconclass.org/{quote(code, safe='')}.json", timeout=30)
            txt = r.json().get("txt", {}) if r.status_code == 200 else {}
            store[code] = txt.get(lang) or txt.get("en") or ""
        except Exception:
            store[code] = ""
        if n % 100 == 0:
            cache.save()
        time.sleep(pause)
    cache.save()
    return {c: store.get(c, "") for c in codes if store.get(c)}


def _pick(links: dict[str, str] | None, lang: str, wiki: dict, max_chars: int):
    """Texte dans la langue du pack, sinon en anglais : (texte, langue) ou None."""
    for wl in (lang, "en"):
        t = (links or {}).get(wl)
        if t:
            text = wiki.get(f"{wl}:{t}", "")
            if len(text) > 80:
                return (_trim(text, max_chars), wl)
    return None


def gather(raw: dict, cache_file: str | Path, wikipedia: bool = True, iconclass: bool = True) -> dict:
    """Rassemble les textes utiles aux fiches pour un pack Wikidata (format 2)."""
    lang = raw.get("lang", "fr")
    cache = Cache(cache_file)
    groups = {
        "art": (raw.get("sitelinks", {}), 1800),
        "creator": ({q: c.get("sitelinks", {}) for q, c in raw.get("creators", {}).items()}, 1500),
        "topic": ({q: t.get("sitelinks", {}) for q, t in raw.get("topics", {}).items()}, 1000),
    }
    out: dict = {}
    if wikipedia:
        titles: dict[str, list[str]] = {}
        for links_by_q, _ in groups.values():
            for links in links_by_q.values():
                for wl, t in (links or {}).items():
                    titles.setdefault(wl, []).append(t)
        wiki_intros(titles, cache)
        wiki = cache.data.get("wiki", {})
        for name, (links_by_q, max_chars) in groups.items():
            out[name] = {q: r for q, links in links_by_q.items() if (r := _pick(links, lang, wiki, max_chars))}
    if iconclass:
        codes = {c for cs in raw.get("iconclass", {}).values() for c in cs}
        out["iconclass"] = iconclass_labels(codes, cache, lang) if codes else {}
    print(f"  textes : {len(out.get('art', {}))} œuvres, {len(out.get('creator', {}))} artistes, "
          f"{len(out.get('topic', {}))} notions, {len(out.get('iconclass', {}))} codes Iconclass", flush=True)
    return out
