# Clara — app Android (Flutter)

Version 0.2 : musées téléchargeables (rangés par pays puis ville), un musée embarqué (SMK), scan par photo
dans le musée choisi ou partout, fiches détaillées, collection et progression par musée.
Internet ne sert qu'à télécharger le catalogue et les packs ; tout le reste fonctionne hors ligne.

## Construction automatique

Chaque push sur `main` lance `.github/workflows/android.yml`, qui :
1. télécharge DINOv2-small (fp32 : la version int8 utilise ConvInteger, absent d'ONNX Runtime sur Android) et construit le pack SMK avec la chaîne Python (`pipeline/`) ;
2. copie modèle et pack dans `assets/` (`tool/prepare_assets.sh`) ;
3. génère le projet Android et le configure (`tool/configure_android.sh` : caméra, minSdk 24, sans Internet) ;
4. compile l'APK arm64. Un tag `v*` publie l'APK dans une release GitHub.

## Construire en local

```bash
# depuis la racine du dépôt, après avoir construit le pack avec le même modèle :
cd pipeline
python -m clara_p0 download-model --variant dinov2-small
python -m clara_p0 build-smk --model models/dinov2-small.onnx --projection none --pooling cls
cd ..
bash app/tool/prepare_assets.sh pipeline/packs/dk-smk-v1 pipeline/models/dinov2-small.onnx
cd app && bash tool/configure_android.sh && flutter run
```

Le pack et l'app doivent utiliser **le même modèle** et le même réglage (CLS, sans projection en v0.1).

## Packs et catalogue

`.github/workflows/packs.yml` construit un pack par musée de `pipeline/museums.json` et `catalog.json`.
À la publication d'une release, ils y sont joints ; l'app lit
`https://github.com/remigl/clara/releases/latest/download/catalog.json`.
Les packs doivent être construits avec le même modèle que l'app (vérifié via `model_id`).

## Code

- `lib/core.dart` : packs (SQLite + vecteurs int8), catalogue et téléchargements, DINOv2 via ONNX Runtime, collection (`user.db`).
- `lib/screens.dart` : Scanner (choix du musée), Ma collection, Musées (pays → ville), fiche d'œuvre.
