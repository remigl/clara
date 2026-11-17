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
# Garantie hors ligne : aucune permission Internet dans l'app publiée
if grep -q 'android.permission.INTERNET' "$M"; then
  sed -i '/android.permission.INTERNET/d' "$M"
fi

for G in android/app/build.gradle.kts android/app/build.gradle; do
  [ -f "$G" ] || continue
  sed -i 's/minSdk = flutter.minSdkVersion/minSdk = 24/; s/minSdkVersion flutter.minSdkVersion/minSdkVersion 24/' "$G"
done
echo "--- AndroidManifest.xml"; cat "$M"
echo "--- minSdk"; grep -h -n "minSdk" android/app/build.gradle* || true
