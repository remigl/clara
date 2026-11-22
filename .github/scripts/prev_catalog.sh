#!/usr/bin/env bash
# Catalogue de la release la plus récente qui en contient un (hors tag exclu).
# Usage : prev_catalog.sh <tag à exclure> <dossier>  ->  <dossier>/catalog.json ; affiche le tag (vide si aucun)
set -uo pipefail
EXCLUDE="${1:-}"; DIR="${2:-prev}"; mkdir -p "$DIR"
for TAG in $(gh api "repos/$GITHUB_REPOSITORY/releases?per_page=30" --jq '.[] | select(.draft==false) | .tag_name'); do
  [ "$TAG" = "$EXCLUDE" ] && continue
  if gh release download "$TAG" -R "$GITHUB_REPOSITORY" -p catalog.json -D "$DIR" --clobber >/dev/null 2>&1; then
    echo "$TAG"; exit 0
  fi
done
echo ""
