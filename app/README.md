# Clara — app Android (Flutter)

Version 0.1 : un musée embarqué (SMK), scan par photo, confirmation, collection, progression, recherche.
Tout fonctionne hors ligne ; l'app n'a pas la permission Internet.

## Construction automatique

Chaque push sur `main` lance `.github/workflows/android.yml`, qui :
1. télécharge DINOv2-small (int8) et construit le pack SMK avec la chaîne Python (`pipeline/`) ;
2. copie modèle et pack dans `assets/` (`tool/prepare_assets.sh`) ;
3. génère le projet Android et le configure (`tool/configure_android.sh` : caméra, minSdk 24, sans Internet) ;
4. compile l'APK arm64. Un tag `v*` publie l'APK dans une release GitHub.

## Construire en local

```bash
# depuis la racine du dépôt, après avoir construit le pack avec le même modèle :
cd pipeline
python -m clara_p0 download-model --variant dinov2-small-int8
python -m clara_p0 build-smk --model models/dinov2-small-int8.onnx --projection none --pooling cls
cd ..
bash app/tool/prepare_assets.sh pipeline/packs/dk-smk-v1 pipeline/models/dinov2-small-int8.onnx
cd app && bash tool/configure_android.sh && flutter run
```

Le pack et l'app doivent utiliser **le même modèle** et le même réglage (CLS, sans projection en v0.1).

## Code

- `lib/core.dart` : pack musée (SQLite + vecteurs int8), DINOv2 via ONNX Runtime, collection (`user.db`).
- `lib/screens.dart` : Scanner, Ma collection, Musée, fiche d'œuvre.
