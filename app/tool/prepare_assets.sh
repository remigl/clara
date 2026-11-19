#!/usr/bin/env bash
# Copie le modèle et un pack musée dans les assets de l'app.
# Usage (depuis la racine du dépôt) : bash app/tool/prepare_assets.sh pipeline/packs/dk-smk-v1 pipeline/models/dinov2-small.onnx
set -euo pipefail
PACK="$1"
MODEL="$2"
DEST="$(dirname "$0")/../assets"
rm -rf "$DEST/packs" "$DEST/model"
mkdir -p "$DEST/packs/dk-smk/thumbs" "$DEST/model"
cp "$PACK/manifest.json" "$PACK/catalog.db" "$PACK/vectors.bin" "$PACK/scales.bin" "$DEST/packs/dk-smk/"
cp "$PACK"/thumbs/*.webp "$DEST/packs/dk-smk/thumbs/"
cp "$MODEL" "$DEST/model/model.onnx"
# l'app v0.1 n'utilise pas encore keypoints.bin (vérification géométrique : prochaine version)
python3 - "$DEST/packs/dk-smk/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"pack {m['id']} v{m['version']} : {m['counts']}, modèle {m['model_id']}, dim {m['embedding_dim']}")
PY
du -sh "$DEST"/model "$DEST"/packs
