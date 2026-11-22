# Signature de l'APK

`clara.keystore` signe tous les APK construits par GitHub Actions, pour qu'une nouvelle version
s'installe **par-dessus** l'ancienne (Android refuse une mise à jour signée avec une autre clé).

La CI le copie à la place de la clé de débogage (`~/.android/debug.keystore`, alias `androiddebugkey`,
mots de passe `android`), que le gabarit Flutter utilise pour les builds release.

Cette clé est publique (dépôt public) : elle garantit la continuité des mises à jour, pas l'authenticité.
Pour une clé secrète, mettre le keystore en base64 dans le secret GitHub `CLARA_KEYSTORE_B64` :
la CI l'utilise alors à la place de celui-ci (même alias et mots de passe).
