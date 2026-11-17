"""Enregistrements normalisés, indépendants de la source (SMK, Wikidata, Europeana…)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

OPEN_RIGHTS = (
    "creativecommons.org/publicdomain/mark/",
    "creativecommons.org/publicdomain/zero/",
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
)
PDM = "http://creativecommons.org/publicdomain/mark/1.0/"


def rights_class(uri: str | None) -> str:
    if not uri:
        return "unknown"
    u = uri.lower()
    if any(k in u for k in OPEN_RIGHTS):
        return "open"
    if "creativecommons.org/licenses/" in u:  # NC / ND
        return "restricted"
    if "rightsstatements.org" in u:
        return "permission" if "inc" in u else "restricted"
    return "unknown"


def normalize_inventory(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"[\s/]+", "-", s)
    return re.sub(r"[^A-Z0-9.\-]", "", s)


@dataclass
class Museum:
    id: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    website: str | None = None
    wikidata_qid: str | None = None
    name_local: str | None = None
    data_quality: int = 0
    geofence_m: int = 250


@dataclass
class Source:
    id: str
    name: str
    url: str
    license: str


@dataclass
class Artwork:
    museum_id: str
    inventory_number: str
    title: str
    source_id: str
    artist_display: str | None = None
    artist_qid: str | None = None
    artist_birth_year: int | None = None
    artist_death_year: int | None = None
    date_display: str | None = None
    date_start: int | None = None
    date_end: int | None = None
    description: str | None = None
    medium: str | None = None
    height_cm: float | None = None
    width_cm: float | None = None
    object_type: str = "painting"
    collection: str | None = None
    room: str | None = None
    floor: str | None = None
    exhibition_status: str = "unknown"
    status_checked_at: str | None = None
    wikidata_qid: str | None = None
    source_url: str | None = None
    image_url: str | None = None
    image_rights_uri: str | None = None
    credit_line: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.museum_id}:{normalize_inventory(self.inventory_number)}"

    @property
    def image_rights_class(self) -> str:
        return rights_class(self.image_rights_uri)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        return d
