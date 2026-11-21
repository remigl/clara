#!/usr/bin/env bash
# Génère le projet Android (flutter create) s'il manque, puis applique la configuration de Clara.
# À lancer depuis app/.
set -euo pipefail
if [ ! -d android ]; then
  flutter create --platforms=android --org be.remigl --project-name clara .
fi
rm -f test/widget_test.dart   # test du gabarit Flutter, sans rapport avec l'app

M=android/app/src/main/AndroidManifest.xml
grep -q 'android.permission.CAMERA' "$M" || \
  sed -i 's#<application#<uses-permission android:name="android.permission.CAMERA"/>\n    <application#' "$M"
sed -i 's/android:label="[^"]*"/android:label="Clara"/' "$M"
# Internet : uniquement pour télécharger le catalogue et les packs des musées (la reconnaissance reste locale)
grep -q 'android.permission.INTERNET' "$M" || \
  sed -i 's#<application#<uses-permission android:name="android.permission.INTERNET"/>\n    <application#' "$M"

for G in android/app/build.gradle.kts android/app/build.gradle; do
  [ -f "$G" ] || continue
  sed -i 's/minSdk = flutter.minSdkVersion/minSdk = 24/; s/minSdkVersion flutter.minSdkVersion/minSdkVersion 24/' "$G"
done
# Version de Kotlin : le gabarit Flutter récent demande une version absente des dépôts (2.3.20 introuvable,
# 21/11/2026). On impose une version publiée, compatible avec l'Android Gradle Plugin du gabarit.
KOTLIN_VERSION="${KOTLIN_VERSION:-2.2.21}"
for S in android/settings.gradle.kts android/settings.gradle; do
  [ -f "$S" ] || continue
  sed -i -E "s/(id\(\"org.jetbrains.kotlin.android\"\) version \")[^\"]+/\1${KOTLIN_VERSION}/" "$S"
  sed -i -E "s/(id \"org.jetbrains.kotlin.android\" version \")[^\"]+/\1${KOTLIN_VERSION}/" "$S"
  echo "--- $S"; cat "$S"
done

echo "--- AndroidManifest.xml"; cat "$M"
echo "--- minSdk"; grep -h -n "minSdk" android/app/build.gradle* || true
