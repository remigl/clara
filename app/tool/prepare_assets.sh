#!/usr/bin/env bash
# Copie le modèle et le pack embarqué (zip) dans les assets de l'app.
# Usage (depuis la racine du dépôt) : bash app/tool/prepare_assets.sh pipeline/packs/dk-smk-v1.zip pipeline/models/dinov2-small.onnx
set -euo pipefail
PACK_ZIP="$1"
MODEL="$2"
DEST="$(dirname "$0")/../assets"
rm -rf "$DEST/packs" "$DEST/model"
mkdir -p "$DEST/packs" "$DEST/model"
cp "$PACK_ZIP" "$DEST/packs/dk-smk.zip"
cp "$MODEL" "$DEST/model/model.onnx"
python3 - "$PACK_ZIP" <<'PY'
import json, sys, zipfile
m = json.loads(zipfile.ZipFile(sys.argv[1]).read("manifest.json"))
print(f"pack embarqué {m['id']} v{m['version']} : {m['counts']}, modèle {m['model_id']}, dim {m['embedding_dim']}")
PY
du -sh "$DEST"/model "$DEST"/packs
