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


MODEL = Path(__file__).resolve().parents[1] / "models" / "dinov2-small.onnx"


@pytest.mark.skipif(not MODEL.exists(), reason="modèle DINOv2 non téléchargé (python -m clara_p0 download-model)")
def test_dinov2_embedder():
    from clara_p0.embed import OnnxDinoEmbedder
    e = OnnxDinoEmbedder(MODEL)
    a = procedural_painting(RNG)
    v = e.embed([a, cv2.flip(a, 1), procedural_painting(RNG)])
    assert v.shape == (3, e.dim) and v[0] @ v[1] > v[0] @ v[2]


def test_sample_queries_keeps_distractors():
    qs = [(f"k{i}", str(i)) for i in range(800)] + [(f"d{i}", None) for i in range(200)]
    s = met.sample_queries(qs, 100)
    assert len(s) == 100 and 5 < sum(t is None for _, t in s) < 40


def test_sweep_on_fake_dataset(tmp_path):
    from clara_p0 import sweep
    test_met_evaluation_on_fake_dataset(tmp_path)  # crée le faux Met et le cache de la base
    rows = sweep.run(ThumbEmbedder(), str(tmp_path / "met"), limit=25, museum_sizes=(0, 10),
                     cache_dir=str(tmp_path / "cache"), out=str(tmp_path / "s.json"))
    assert len(rows) == 3 * 2 * 2 and all(0 <= r["top1"] <= 1 for r in rows)
    assert sweep.coverage_at_far(np.array([.9, .8, .7]), np.array([True, True, False]), np.array([True] * 3)) == 2 / 3
