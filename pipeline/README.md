# Clara — pipeline P0

Preuve de concept en Python de la reconnaissance et de la construction des packs.
C'est la référence de ce que fera l'app sur le téléphone : mêmes étapes, même format de pack.

Tout est libre et gratuit : DINOv2 (Apache 2.0), OpenCV/SIFT, ONNX Runtime (MIT), données CC0 / domaine public.

## Installation (Linux ou Windows, Python 3.10+)

```bash
cd pipeline
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests          # 12 tests, ~15 s, sans réseau
```

## 1. Vérifier que tout marche (hors ligne, 1 min)

```bash
python -m clara_p0 selftest
```

Fausses peintures, dont des « séries » quasi identiques, photographiées par un simulateur de musée
(cadre, perspective, reflet, visiteur devant, flou, JPEG). Le modèle par défaut est une vignette 16×16 :
c'est un plancher volontairement faible, pas le vrai modèle.

## 2. Télécharger le modèle (88 Mo)

```bash
python -m clara_p0 download-model                     # DINOv2-small fp32
python -m clara_p0 selftest --model models/dinov2-small.onnx
```

## 3. Mesurer sur de vraies photos de visiteurs : le Met Dataset

```bash
python -m clara_p0 download-met                       # ~5 Go : base mini + requêtes + vérité terrain
python -m clara_p0 eval-met --limit 200               # essai rapide
python -m clara_p0 eval-met                           # complet -> results/met.json
```

- La base est encodée une seule fois, avec un cache reprenable (`cache/`, Ctrl+C sans perte).
- `valset` calibre la confiance (sauvée dans `cache/calibration_*.json`), `testset` mesure.
- Le rapport compare **embedding seul** et **embedding + vérification géométrique + confiance**, avec
  top-1, GAP (la métrique officielle du Met), fausses acceptations, couverture et rejet des objets hors base.

Améliorer avec une projection 256-d (aussi la taille cible dans l'app) :

```bash
python -m clara_p0 fit-projection
python -m clara_p0 eval-met --projection models/projection-256.npz
```

Attention : le Met contient tous types d'objets (vases, armures, monnaies…), plus difficiles que des peintures.
Les chiffres sur des tableaux seront meilleurs.

## 4. Construire un pack musée

```bash
python -m clara_p0 build-smk                          # SMK Copenhague : peintures exposées, salle connue
python -m clara_p0 build-smk --include-not-on-display # toute la collection domaine public
python -m clara_p0 build-wikidata --qid Q23402 --id fr-orsay   # n'importe quel musée, niveau 0
```

Sortie : `packs/<musée>-v1/` + `.zip` (`catalog.db`, `vectors.bin`, `scales.bin`, `keypoints.bin`,
`thumbs/`, `manifest.json`). Les métadonnées brutes restent dans `cache/` (`--offline-raw` pour reconstruire
sans retélécharger).

Politique de droits v1 : seules les images « open » (domaine public, CC0, CC BY, CC BY-SA) sont encodées et
affichées ; les autres œuvres sont au catalogue sans image. Pour Wikidata, une image est considérée domaine
public si l'artiste est mort depuis plus de 70 ans.

## 5. Reconnaître une photo avec un pack

```bash
python -m clara_p0 recognize --pack packs/dk-smk-v1 ma_photo.jpg
```

Utilise le même modèle (et la même projection) que pour construire le pack.

## Organisation du code

| Module | Rôle |
|---|---|
| `views.py` | détection du tableau (contours, P0), redressement, vues de référence |
| `embed.py` | DINOv2 ONNX (CLS + GeM des patchs), projection PCA-whitening, référence triviale |
| `index.py` | index exact int8, format `vectors.bin` de l'app |
| `verify.py` | vérification géométrique SIFT + RANSAC (XFeat dans l'app) |
| `recognize.py` | chaîne complète, confiance calibrée, décision, vote temporel |
| `metrics.py`, `evaluate.py`, `met.py`, `selftest.py` | évaluation |
| `sources/smk.py`, `sources/wikidata.py` | ingestion des données |
| `pack.py` | construction et lecture des packs |
| `synth.py` | générateur de photos de musée (servira aussi à l'entraînement en P2) |

## Limites connues de P0

- La détection du tableau est classique (contours) ; le détecteur appris « 4 coins » arrive en P2.
- Les champs de l'API SMK sont lus de façon défensive mais n'ont pas pu être testés contre l'API réelle
  pendant le développement : la réponse brute est gardée dans `cache/smk/raw.jsonl` pour corriger un champ.
- La latence affichée est celle du PC, pas du téléphone.
