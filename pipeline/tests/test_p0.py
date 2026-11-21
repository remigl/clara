import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from clara_p0 import met, pack
from clara_p0.embed import Projection, ThumbEmbedder, l2n
from clara_p0.imageio import write_image
from clara_p0.index import FlatIndex, quantize
from clara_p0.metrics import gap, summarize
from clara_p0.recognize import Calibrator, Recognizer, temporal_vote
from clara_p0.records import PDM, Artwork, Museum, Source, normalize_inventory, rights_class
from clara_p0.sources import smk, wikidata
from clara_p0.synth import museum_photo, procedural_painting
from clara_p0.verify import GeometricVerifier, Keypoints, KeypointStore

RNG = np.random.default_rng(0)


def test_quantize_roundtrip():
    x = l2n(RNG.standard_normal((50, 256)).astype(np.float32))
    q, s = quantize(x)
    err = np.abs(q.astype(np.float32) * s[:, None] - x).max()
    assert err < 0.01


def test_index_search_and_io(tmp_path):
    x = l2n(RNG.standard_normal((30, 64)).astype(np.float32))
    idx = FlatIndex(64)
    idx.add(x, [(f"a{i // 3}", i % 3, "full") for i in range(30)])
    top = idx.search(x[7] + 0.01, k=3)[0]
    assert top[0][0] == "a2" and len(top) == 3
    idx.save(tmp_path)
    idx2 = FlatIndex.load(tmp_path)
    assert idx2.search(x[7], k=1)[0][0][0] == "a2"


def test_projection_reduces_dim():
    x = RNG.standard_normal((500, 64))
    p = Projection.fit(x, dim=16)
    y = p(x.astype(np.float32))
    assert y.shape == (500, 16) and np.allclose(np.linalg.norm(y, axis=1), 1, atol=1e-4)


def test_keypoints_and_verifier():
    v = GeometricVerifier()
    a = procedural_painting(RNG)
    b = procedural_painting(RNG)
    ka = v.extract(a)
    assert Keypoints.from_bytes(ka.to_bytes()).desc.shape == ka.desc.shape
    photo = museum_photo(a, np.random.default_rng(3), strength=0.5)
    kq = v.extract(photo)
    assert v.inliers(kq, ka) > 3 * max(v.inliers(kq, v.extract(b)), 5)


def test_metrics():
    pred = ["a", "b", "x", "c"]
    truth = ["a", "b", None, "z"]
    conf = [0.99, 0.95, 0.2, 0.1]
    assert gap(pred, conf, truth) == pytest.approx(2 / 3)
    s = summarize(pred, conf, truth, accept=0.9, confirm=0.5)
    assert s["top1_accuracy"] == pytest.approx(2 / 3)
    assert s["false_accept_rate_auto"] == 0 and s["distractor_rejection"] == 1


def test_calibrator_learns():
    feats = [{"cos1": c, "cos_margin": c / 5, "inl1": c * 50, "inl_margin": c * 40, "det": 0.5}
             for c in np.linspace(0, 1, 200)]
    ok = [f["cos1"] > 0.5 for f in feats]
    cal = Calibrator.fit(feats, ok)
    assert cal.predict(feats[-1]) > 0.9 and cal.predict(feats[0]) < 0.1


def test_temporal_vote():
    class R:
        def __init__(s, b, c): s.best, s.confidence = b, c
    assert temporal_vote([R("a", .8), R("a", .7), R("b", .9), R("a", .6)]) == "a"
    assert temporal_vote([R("a", .8), R("b", .7)]) is None


def test_rights_and_ids():
    assert rights_class(PDM) == "open"
    assert rights_class("https://creativecommons.org/licenses/by-nc/4.0/") == "restricted"
    assert rights_class(None) == "unknown"
    assert normalize_inventory(" kms 1/2 ") == "KMS-1-2"


def _synthetic_museum(tmp_path, n=40):
    refs, arts = {}, []
    for i in range(n):
        img = procedural_painting(RNG)
        p = tmp_path / "src" / f"{i}.png"
        write_image(p, img)
        a = Artwork(museum_id="xx-test", inventory_number=f"INV {i}", title=f"Tableau {i}", source_id="test",
                    artist_display="Peintre Test" if i % 2 else "Autre Artiste", image_url=str(p),
                    image_rights_uri=PDM if i < n - 3 else None, room=f"Salle {i % 5}",
                    exhibition_status="on_display" if i % 10 else "not_on_display", collection="Peintures")
        refs[a.id] = img
        arts.append(a)
    museum = Museum(id="xx-test", name="Musée test", city="Ville", country="BE", lat=50.0, lon=4.0, data_quality=3)
    return museum, arts, refs


def test_pack_build_load_recognize(tmp_path):
    museum, arts, refs = _synthetic_museum(tmp_path)
    emb = ThumbEmbedder()
    d = pack.build(museum, [Source("test", "Test", "", "CC0")], arts, emb, tmp_path / "packs", tmp_path / "cache")
    assert (d.with_suffix(".zip")).exists()
    p = pack.Pack(d)
    assert p.verify_integrity() == []
    m = p.manifest
    assert m["counts"]["artworks"] == 40
    assert m["counts"]["recognizable"] == 37           # 3 images aux droits inconnus : pas encodées
    assert m["counts"]["progress_denominator"] == 36   # 4 œuvres « not_on_display »
    assert m["counts"]["with_thumbnail"] == 37
    assert p.search_text("autre artiste", 50) and len(p.search_text("autre", 50)) == 20

    rec = Recognizer(emb, p.index, p.keypoints)
    aid = arts[5].id
    r = rec.recognize(museum_photo(refs[aid], np.random.default_rng(1), strength=0.4))
    assert r.best == aid

    # progression : même requête que l'app (ARCHITECTURE.md / schema.sql)
    user = tmp_path / "user.db"
    u = sqlite3.connect(user)
    u.executescript(pack.user_schema())
    u.execute("INSERT INTO collected VALUES (?,?,?,?,?,?)", (aid, "xx-test", "t", "t", 1, "scan"))
    u.execute("ATTACH ? AS c", (str(d / "catalog.db"),))
    (prog,) = u.execute("""SELECT COUNT(*) * 1.0 / m.number_of_works FROM collected uc
        JOIN c.artwork a ON a.id = uc.artwork_id JOIN c.museum m ON m.id = a.museum_id
        WHERE a.museum_id = 'xx-test' AND a.counts_for_progress = 1""").fetchone()
    assert prog == pytest.approx(1 / 36)


def test_smk_mapping():
    item = {"object_number": "KMS1", "titles": [{"title": "Titanernes fald", "language": "da"},
                                                {"title": "The Fall of the Titans", "language": "en"}],
            "production": [{"creator": "Cornelis Cornelisz. van Haarlem", "creator_date_of_death": "1638-11-11"}],
            "production_date": [{"start": "1588-01-01T00:00:00Z", "end": "1590-12-31", "period": "1588-1590"}],
            "object_names": [{"name": "Maleri"}], "current_location_name": "Sal 205", "public_domain": True,
            "has_image": True, "image_thumbnail": "https://example.org/kms1.jpg",
            "dimensions": [{"type": "højde", "value": "246", "unit": "cm"}, {"type": "bredde", "value": "311"}]}
    a = smk.to_artwork(item)
    assert a.id == "dk-smk:KMS1" and a.title == "The Fall of the Titans"
    assert a.exhibition_status == "on_display" and a.room == "Sal 205"
    assert a.image_rights_class == "open" and a.date_start == 1588 and a.height_cm == 246
    stored = dict(item, current_location_name=None)
    assert smk.select([item, stored]) == [a] or len(smk.select([item, stored])) == 1
    assert smk.to_artwork(stored).exhibition_status == "not_on_display"


def test_wikidata_mapping():
    b = lambda **kw: {k: {"value": v} for k, v in kw.items()}  # noqa: E731
    rows = [b(item="http://www.wikidata.org/entity/Q1", itemLabel="Le Déjeuner", inv="RF 1986", death="1883-04-30T00:00:00Z",
              creatorLabel="Édouard Manet", image="http://commons.wikimedia.org/wiki/Special:FilePath/X.jpg"),
            b(item="http://www.wikidata.org/entity/Q1", itemLabel="Le Déjeuner", inv="RF 1986", creatorLabel="Doublon"),
            b(item="http://www.wikidata.org/entity/Q2", itemLabel="Q2", death="1990-01-01T00:00:00Z")]
    arts = wikidata.select(rows, "fr-test")
    assert len(arts) == 2
    assert arts[0].id == "fr-test:RF-1986" and arts[0].image_rights_class == "open"
    assert arts[0].image_url.startswith("https://") and arts[0].image_url.endswith("?width=1024")
    assert arts[1].image_rights_class == "unknown" and arts[1].title == "Sans titre"


def test_met_evaluation_on_fake_dataset(tmp_path):
    root = tmp_path / "met"
    gt = root / "ground_truth"
    gt.mkdir(parents=True)
    db, val, test = [], [], []
    rng = np.random.default_rng(5)
    for i in range(20):
        img = procedural_painting(rng)
        write_image(root / "images" / f"MET/{i}/0.jpg", img)
        db.append({"path": f"MET/{i}/0.jpg", "id": i})
        for split, lst in (("val", val), ("test", test)):
            write_image(root / "images" / f"q/{split}{i}.jpg", museum_photo(img, rng, strength=0.3))
            lst.append({"path": f"q/{split}{i}.jpg", "MET_id": i})
    for split, lst in (("val", val), ("test", test)):
        for j in range(5):
            write_image(root / "images" / f"d/{split}{j}.jpg", museum_photo(procedural_painting(rng), rng))
            lst.append({"path": f"d/{split}{j}.jpg"})
    (gt / "mini_MET_database.json").write_text(json.dumps(db))
    (gt / "valset.json").write_text(json.dumps(val))
    (gt / "testset.json").write_text(json.dumps(test))
    res = met.evaluate(ThumbEmbedder(), str(root), cache_dir=str(tmp_path / "cache"), out=str(tmp_path / "r.json"))
    assert res["with_rerank"]["n_known"] == 20 and res["with_rerank"]["n_distractors"] == 5
    assert res["with_rerank"]["top1_accuracy"] >= 0.6
    assert (tmp_path / "r.json").exists()
    res = met.evaluate(ThumbEmbedder(), str(root), cache_dir=str(tmp_path / "cache"), museum_size=12)
    assert "safe_coverage_far1" in res["with_rerank"] and res["museum_size"] == 12


MODEL = Path(__file__).resolve().parents[1] / "models" / "dinov2-small.onnx"


@pytest.mark.skipif(not MODEL.exists(), reason="modèle DINOv2 non téléchargé (python -m clara_p0 download-model)")
def test_dinov2_embedder():
    from clara_p0.embed import OnnxDinoEmbedder
    e = OnnxDinoEmbedder(MODEL)
    a = procedural_painting(RNG)
    v = e.embed([a, cv2.flip(a, 1), procedural_painting(RNG)])
    assert v.shape == (3, e.dim) and v[0] @ v[1] > v[0] @ v[2]


def test_balanced_queries():
    qs = [(f"k{i}", str(i)) for i in range(100)] + [(f"d{i}", None) for i in range(2000)]
    s = met.balanced_queries(qs, None, 300)
    assert sum(t is not None for _, t in s) == 100 and sum(t is None for _, t in s) == 300
    s = met.balanced_queries(qs, 10, 5, keep=lambda t: int(t) % 2 == 0)
    assert all(int(t) % 2 == 0 for _, t in s if t) and len(s) == 15
    assert met.is_painting("European Paintings|") and met.is_painting("Asian Art|Paintings")
    assert not met.is_painting("Egyptian Art|Sculpture") and not met.is_painting(None)


def test_sweep_on_fake_dataset(tmp_path):
    from clara_p0 import sweep
    test_met_evaluation_on_fake_dataset(tmp_path)  # crée le faux Met et le cache de la base
    rows = sweep.run(ThumbEmbedder(), str(tmp_path / "met"), distractors=5, museum_sizes=(0, 10),
                     cache_dir=str(tmp_path / "cache"), out=str(tmp_path / "s.json"))
    assert len(rows) == 3 * 2 * 2 and all(0 <= r["top1"] <= 1 for r in rows)
    assert sweep.coverage_at_far(np.array([.9, .8, .7]), np.array([True, True, False]), np.array([True] * 3)) == 2 / 3


def test_decision_rule():
    from clara_p0.recognize import DecisionRule
    feats = [{"inl1": float(i), "cos1": 0.5} for i in range(60)]
    ok = [i >= 30 for i in range(60)]
    r = DecisionRule.fit(feats, ok)
    assert 29 < r.t_auto <= 31 and r.predict(feats[45]) >= 0.9 and r.predict(feats[5]) < 0.5
    assert r.t_confirm <= r.t_auto
    assert DecisionRule(20, 10).predict({"inl1": 15, "cos1": 0}) == pytest.approx(0.7)


def test_legacy_cache_derivation(tmp_path):
    """Un cache 'cls+gem' permet d'obtenir l'empreinte 'cls' sans réencoder."""
    class Fake(ThumbEmbedder):
        def __init__(self):
            super().__init__()
            self.legacy_id, self.raw_id = "m", "m-cls"
        def from_legacy(self, v):
            return l2n(v[:, : v.shape[1] // 2])
        def embed_raw(self, images):
            raise AssertionError("ne doit pas réencoder")
    legacy = l2n(RNG.standard_normal((5, 8)).astype(np.float32))
    np.savez(tmp_path / "met_mini_m.npz", vectors=legacy, n=5)
    x = met.raw_database(Fake(), None, [("p", str(i)) for i in range(5)], tmp_path / "met_mini_m-cls.npz")
    assert x.shape == (5, 4) and np.allclose(x, l2n(legacy[:, :4]))


def test_smk_fetch_retries(tmp_path, monkeypatch):
    import requests
    calls = {"n": 0}

    class R:
        def __init__(self, code, data=None):
            self.status_code, self._d, self.reason = code, data, "x"
        def raise_for_status(self):
            pass
        def json(self):
            return self._d

    def fake_get(self, url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return R(503)
        if calls["n"] == 2:
            raise requests.exceptions.ChunkedEncodingError("coupé")
        off = params["offset"]
        return R(200, {"found": 3, "items": [{"object_number": f"K{off + i}"} for i in range(min(2, 3 - off))]})

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(smk.time, "sleep", lambda s: None)
    items = smk.fetch(tmp_path, rows=2)
    assert [i["object_number"] for i in items] == ["K0", "K1", "K2"]


def test_wikidata_details_and_catalog(tmp_path):
    from clara_p0 import catalog
    b = lambda **kw: {k: {"value": v} for k, v in kw.items()}  # noqa: E731
    raw = {"main": [b(item="http://www.wikidata.org/entity/Q1", itemLabel="La Joconde", inv="INV 779",
                      itemDescription="tableau de Léonard de Vinci", death="1519-05-02T00:00:00Z", birth="1452-04-15T00:00:00Z",
                      creatorLabel="Léonard de Vinci", image="http://commons.wikimedia.org/wiki/Special:FilePath/M.jpg"),
                    b(item="http://www.wikidata.org/entity/Q2", itemLabel="Autre", inv="RF 1",
                      itemDescription="peinture")],
           "props": {"Genre": {"Q1": ["portrait"]}, "Matériaux": {"Q1": ["peinture à l'huile", "bois de peuplier"]},
                     "_nationality": {"Q1": ["république de Florence"]}},
           "sitelinks": {"Q1": {"fr": "La Joconde"}}, "lang": "fr"}
    arts = wikidata.select(raw, "fr-test", summaries={"Q1": ("La Joconde est un portrait…" * 3, "fr")})
    a = {x.id: x for x in arts}["fr-test:INV-779"]
    assert a.description.startswith("La Joconde") and "Wikipédia (fr)" in a.description_credit
    assert ("Genre", "portrait") in a.facts and a.medium == "peinture à l'huile, bois de peuplier"
    assert a.artist_nationality == "république de Florence" and a.artist_life == "1452–1519"
    assert {x.id: x for x in arts}["fr-test:RF-1"].description == "Peinture."

    museum, arts2, _ = _synthetic_museum(tmp_path, n=12)
    museum.country_name = "Belgique"
    arts2[0].description, arts2[0].facts = "Un texte.", [("Genre", "paysage")]
    d = pack.build(museum, [Source("test", "Test", "", "CC0")], arts2, ThumbEmbedder(), tmp_path / "packs", tmp_path / "cache")
    row = sqlite3.connect(d / "catalog.db").execute(
        "SELECT description, facts_json FROM artwork WHERE id = ?", (arts2[0].id,)).fetchone()
    assert row[0] == "Un texte." and json.loads(row[1]) == [["Genre", "paysage"]]
    cat = json.loads(catalog.merge(str(tmp_path / "packs")).read_text())
    p = cat["packs"][0]
    assert p["id"] == "xx-test" and p["country_name"] == "Belgique" and p["file"] == "xx-test-v1.zip" and p["bytes"] > 0


def test_wikidata_format2_enrichment_and_pack(tmp_path):
    from clara_p0.sources import knowledge
    b = lambda **kw: {k: {"value": v} for k, v in kw.items()}  # noqa: E731
    raw = {
        "format": 2, "lang": "fr",
        "main": [b(item="http://www.wikidata.org/entity/Q10", itemLabel="La Ronde de nuit", inv="SK-C-5",
                   creator="http://www.wikidata.org/entity/Q5598", creatorLabel="Rembrandt",
                   birth="1606-07-15T00:00:00Z", death="1669-10-04T00:00:00Z",
                   image="http://commons.wikimedia.org/wiki/Special:FilePath/N.jpg")],
        "props": {"Mouvement": {"Q10": [["Q37853", "baroque"]]}, "Genre": {"Q10": [["Q2", "portrait de groupe"]]},
                  "Commanditaire": {"Q10": [["Q3", "compagnie de Frans Banninck Cocq"]]},
                  "Expositions": {"Q10": [["Q4", "Rembrandt 400"]]}},
        "iconclass": {"Q10": ["45C"]},
        "provenance": {"Q10": [["Kloveniersdoelen", 1642, 1715], ["Rijksmuseum", 1808, None]]},
        "sitelinks": {"Q10": {"fr": "La Ronde de nuit"}},
        "creators": {"Q5598": {"birth_place": [["Q1", "Leyde"]], "death_place": [["Q9", "Amsterdam"]],
                               "teachers": [["Q7", "Pieter Lastman"]], "movements": [["Q37853", "baroque"]],
                               "sitelinks": {"fr": "Rembrandt"}}},
        "topics": {"Q37853": {"label": "baroque", "sitelinks": {"fr": "Baroque"}}, "Q2": {"label": "portrait de groupe"}},
    }
    know = {"art": {"Q10": ("La Ronde de nuit est un tableau…", "fr")},
            "creator": {"Q5598": ("Rembrandt est un peintre…", "fr")},
            "topic": {"Q37853": ("Le baroque est un mouvement…", "fr")}, "iconclass": {"45C": "armes"}}
    a = wikidata.select(raw, "nl-test", know=know)[0]
    secs = dict(a.sections)
    assert ("Iconclass", "armes") in secs["Iconographie"]
    assert secs["Provenance"][0] == ("1642–1715", "Kloveniersdoelen") and secs["Provenance"][1] == ("depuis 1808", "Rijksmuseum")
    assert ("Commanditaire", "compagnie de Frans Banninck Cocq") in secs["Création"]
    assert a.artist_bio.startswith("Rembrandt") and a.artist_birth_place == "Leyde" and a.artist_teachers == "Pieter Lastman"
    assert [g[0] for g in a.glossary] == ["baroque"]

    # croisement par inventaire (ex. API SMK)
    other = Artwork(museum_id="nl-test", inventory_number="SK-C-5", title="The Night Watch", source_id="x",
                    description="Texte du musée.", description_credit="Musée")
    assert wikidata.enrich_by_inventory([other], raw, know, "nl-test") == 1
    assert other.description == "Texte du musée." and other.artist_bio and other.wikidata_qid == "Q10"
    assert dict(other.sections)["Sur Wikipédia"][0][1].startswith("La Ronde")

    # pack : artiste, glossaire, sections
    museum, arts, _ = _synthetic_museum(tmp_path, n=6)
    arts[0].artist_qid, arts[0].artist_bio, arts[0].glossary = "Q5598", "Bio.", [("baroque", "Le baroque…", "Wikipédia")]
    arts[0].sections = [("Provenance", [("1642–1715", "X")])]
    d = pack.build(museum, [Source("test", "Test", "", "CC0")], arts, ThumbEmbedder(), tmp_path / "p", tmp_path / "c")
    db = sqlite3.connect(d / "catalog.db")
    assert db.execute("SELECT bio FROM artist WHERE id='Q5598'").fetchone()[0] == "Bio."
    assert db.execute("SELECT term FROM artwork_glossary WHERE artwork_id=?", (arts[0].id,)).fetchone()[0] == "baroque"
    assert json.loads(db.execute("SELECT sections_json FROM artwork WHERE id=?", (arts[0].id,)).fetchone()[0]) == \
        [["Provenance", [["1642–1715", "X"]]]]
    assert knowledge._trim("a" * 50 + "\n" + "b" * 50 + "\n" + "c" * 50, 120) == "a" * 50 + "\n\n" + "b" * 50
    assert "VALUES ?type { wd:Q3305213 wd:Q93184 }" in wikidata.scope("Q1", "both", ("painting", "drawing"))
