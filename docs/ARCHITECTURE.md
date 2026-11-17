# Clara — Architecture

> Scanner un tableau dans un musée, le reconnaître sur l'appareil, l'ajouter à sa collection et voir sa progression dans le musée.
> **Contrainte n°1 : 100 % hors ligne** une fois le pack du musée téléchargé. Aucune photo ne quitte le téléphone.

---

## 0. Décisions clés (résumé)

| Sujet | Choix | Pourquoi |
|---|---|---|
| App | **Flutter** (Android d'abord, iOS ensuite) + cœur natif C++ via FFI | une base de code ; l'inférence et la recherche passent par du natif, sans copie d'image en Dart |
| Inférence | **ONNX Runtime Mobile** (XNNPACK sur Android, Core ML EP sur iOS) | un seul format de modèle pour les deux OS ; LiteRT (TFLite) en plan B si le GPU Android devient nécessaire |
| Détection du tableau | petit modèle « 4 coins » (MobileNetV3 + heatmaps) entraîné sur **données synthétiques** | les méthodes classiques (contours, Hough) échouent sur les cadres dorés, les reflets et les visiteurs devant le tableau |
| Embedding | **DINOv2 ViT-S/14** (Apache 2.0) comme base, **fine-tuné en metric learning** sur les images de référence augmentées « façon photo de musée », tête 256-d | on reconnaît *ce tableau précis*, pas *un tableau de ce genre* : DINO est meilleur que CLIP pour ça |
| Reranking | **XFeat** (Apache 2.0) + RANSAC homographie sur le top-5 | sépare deux Nymphéas de Monet ; donne une confiance fondée sur la géométrie |
| Recherche vectorielle | **brute force int8 en mémoire** pour un musée ; **USearch HNSW** (Apache 2.0, mmap) pour « Reconnaître partout » | un musée tient en quelques Mo : l'exactitude ne coûte rien. Un index ANN ne vaut la peine qu'au-delà de 100 000 œuvres |
| Stockage | SQLite (`catalog.db` lecture seule par pack + `user.db`) + fichier `vectors.bin` mmap | les données utilisateur ne sont jamais écrasées par une mise à jour de catalogue |
| Données | Wikidata + Europeana + API/open data des musées, **pipeline de build hors appareil** qui produit des packs statiques | aucun serveur applicatif : les packs sont de simples fichiers (GitHub Releases ou n'importe quel CDN) |

---

## 1. Vue d'ensemble

```
                  ┌──────────────── HORS APPAREIL (ton PC, une fois par build) ───────────────┐
 Wikidata ───┐    │  ingest → normalisation → fusion → droits → statut d'exposition          │
 Europeana ──┼──▶ │  → téléchargement des images → embeddings + keypoints → contrôle qualité  │──▶ packs/*.zip
 API musées ─┘    └──────────────────────────────────────────────────────────────────────────┘   (statiques)
                                                                                                    │
                                                                       téléchargement unique (Wi-Fi)│
 ┌──────────────────────────────────── SUR L'APPAREIL, 100 % OFFLINE ─────────────────────────────▼──┐
 │ Caméra ─▶ Détection 4 coins ─▶ Rectification ─▶ Embedding ─▶ Recherche (musée courant) ─▶ top-5     │
 │                                                                                         │          │
 │                                     Vote temporel (plusieurs images) ◀── Reranking XFeat ◀┘        │
 │                                                   │                                                │
 │                                        Décision (accepter / confirmer / rien)                       │
 │                                                   ▼                                                │
 │                       user.db : collection, historique, favoris, notes ─▶ progression              │
 └────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Seule étape en ligne : **télécharger un pack** (et ses mises à jour), idéalement avant la visite. Tout le reste fonctionne en mode avion.

---

## 2. Stack applicative

- **Flutter / Dart** pour l'interface, la navigation, la collection et les statistiques.
- **`clara_core`** : bibliothèque C++ (FFI) qui contient :
  - le prétraitement (rectification perspective, normalisation) — OpenCV *minimal* (core + imgproc uniquement, ~3 Mo) ;
  - l'inférence ONNX Runtime (détecteur, encodeur, XFeat) ;
  - la recherche vectorielle (produit scalaire int8 NEON / `dotprod`) et USearch ;
  - le reranking (matching + RANSAC).
- Les images caméra (YUV) passent directement de la couche native caméra à `clara_core` : **aucune copie d'image en Dart**, et un isolate dédié pour ne pas bloquer l'interface.
- SQLite via `sqlite3` (FFI) avec FTS5 pour la recherche plein texte hors ligne.

Pourquoi pas tout en Dart ? Une recherche sur 30 000 vecteurs et un RANSAC à chaque image en Dart pur ralentiraient la visée en temps réel. Le C++ partagé fonctionne aussi tel quel sur iOS.

---

## 3. Données

### 3.1 Sources (toutes réutilisables, à vérifier musée par musée)

| Source | Ce qu'elle apporte | Licence |
|---|---|---|
| **Wikidata** | colonne vertébrale : œuvre ↔ musée (P195), n° d'inventaire (P217), lieu (P276), artiste, dates, image Commons (P18) | CC0 |
| **Wikimedia Commons** | images haute définition d'œuvres du domaine public | PD / CC, par fichier |
| **Europeana** (Search/Record API, `reusability=open`) | métadonnées agrégées, liens vers les fournisseurs, droits par objet | métadonnées CC0, droits images par objet |
| **POP / Joconde** (Ministère de la Culture) | collections des musées de France | Licence Ouverte Etalab |
| **API / open data des musées** : SMK (Copenhague), Rijksmuseum, Nationalmuseum (Stockholm), Paris Musées, Belvedere (Vienne, Open Content)… | la meilleure source pour le **statut d'exposition** et la **salle** quand elle existe | souvent CC0 / CC BY-SA, à vérifier |

Règle : **les métadonnées et les images n'ont pas les mêmes droits**. On stocke `rights_uri` image par image (`artwork_image`) et on n'embarque une vignette que si elle est classée `open` (PDM, CC0, CC BY, CC BY-SA).

### 3.2 Droits : trois cas distincts

1. **Image affichable** (domaine public ou licence ouverte) → vignette dans la fiche, avec crédit.
2. **Œuvre encore protégée** (artiste mort il y a moins de 70 ans) ou image sans droits clairs → fiche **sans image** (titre, artiste, date) ; l'utilisateur voit *sa propre photo*, qui reste sur son téléphone.
3. **Embeddings calculés à partir d'images protégées** : c'est un point juridique à faire valider (exception de fouille de textes et de données, art. 4 de la directive UE 2019/790, sauf opposition du titulaire). En attendant, la v1 se limite aux images `open`. Je ne suis pas juriste : avant de publier, fais relire ce point.

### 3.3 Statut d'exposition : niveaux de qualité par musée

Ne jamais supposer qu'une œuvre de la collection est accrochée. Chaque musée a un **niveau de données** (`museum.data_quality`) :

| Niveau | Contenu | Source typique | Dénominateur de progression |
|---|---|---|---|
| 0 | appartient à la collection | Wikidata / Europeana | « collection Clara » (libellé explicite) |
| 1 | collection, filtrée par type (peintures) et par des signaux de visibilité (P276 = bâtiment du musée, œuvres « phares ») | Wikidata + heuristiques | idem, plus réaliste |
| 2 | indicateur « exposé / non exposé » | API du musée | **œuvres exposées** |
| 3 | salle et étage | API / plans du musée | exposées, avec progression par salle |
| 4 | flux vérifié et daté (prêts, absences temporaires) | partenariat avec le musée | exposées, à jour |

Chaque œuvre porte `exhibition_status` (`on_display | temporarily_absent | in_storage | on_loan | unknown`), la source du statut et **la date de vérification**. L'interface affiche « vu en salle 34 » quand on le sait, et rien quand on ne le sait pas.

Quand l'utilisateur scanne une œuvre marquée `in_storage`, on l'ajoute quand même à sa collection : c'est la preuve qu'elle est accrochée. Ce signal reste **local** ; un partage anonyme et volontaire pourra venir plus tard, en option, jamais pour une fonction essentielle.

Pour les musées de niveau 0–1, on compte la progression sur un sous-ensemble défini (`counts_for_progress = 1`, par exemple les peintures seulement), sinon personne ne dépasse 3 % au Louvre.

### 3.4 Identifiants stables

`artwork.id = <museumId>:<n° d'inventaire normalisé>` (ex. `fr-orsay:RF-1986-3`). Le QID Wikidata est stocké à part. Si un build doit renommer un id, `id_alias` dans `user.db` assure la migration : **une mise à jour de catalogue ne fait jamais perdre une œuvre collectionnée**.

### 3.5 Modèle

Schéma complet : [`data/schema.sql`](../data/schema.sql). Ce qui s'écarte de ta proposition :

- `visualEmbedding` n'est pas une colonne de `artwork` : c'est une table `artwork_embedding` avec **plusieurs vues par œuvre** (image complète + recadrages) qui pointe vers une ligne de `vectors.bin`, lié à un `model_id`. Changer de modèle oblige à régénérer les vecteurs, et la base le sait. Les vues ajoutées après confirmation de l'utilisateur vont dans `user_embedding` (`user.db`).
- `image` + `imageRights` deviennent `artwork_image` (chemin local, droits, crédit, URL d'origine).
- `scannedWorks` / `scanHistory` sont séparés : `scan` (chaque tentative, avec le top-5 et la décision) et `collected` (une ligne par œuvre, premier et dernier passage, nombre de passages). L'historique sert au diagnostic local et à améliorer les seuils.
- `artwork_keypoints` stocke les descripteurs XFeat de référence pour le reranking.

### 3.6 Format d'un pack

```
fr-orsay-v7.zip
├── manifest.json        # id, version, model_id, dim, sha256, taille, niveau de données
├── catalog.db           # métadonnées SQLite + FTS5
├── vectors.bin          # int8[n_vues × 256], aligné, mmap ; index → (artwork_id, view_idx)
├── keypoints.bin        # XFeat, 128 points × (x, y, desc f16[64]) ≈ 17 Ko/œuvre
└── thumbs/*.webp        # 512 px, uniquement les images « open »
```

Ordre de grandeur pour un musée de 1 500 peintures : vecteurs 1,2 Mo, keypoints ~25 Mo, vignettes ~60 Mo. On peut proposer un **pack léger** sans vignettes (~30 Mo).

---

## 4. Reconnaissance

### 4.1 Le vrai problème

Ce n'est pas de la classification (« c'est un paysage impressionniste ») mais de la **recherche d'instance** : retrouver *exactement* l'image de référence d'un tableau parmi N, à partir d'une photo déformée. Les pièges sont :

- **les œuvres quasi identiques** : séries (Nymphéas, Meules, Cathédrales de Rouen), répliques d'atelier, études préparatoires ;
- **l'écart de domaine** : la référence est un scan propre, la requête une photo inclinée avec un reflet et une tête devant ;
- **les œuvres hors base** : sculptures, cartels, œuvres d'expositions temporaires. Le système doit savoir dire « je ne sais pas ».

### 4.2 Pipeline

```
image caméra (≈ 5 i/s en mode visée)
 → [1] détection des 4 coins du tableau   (MobileNetV3-small, 256×256, ~8 ms)
 → [2] homographie → image rectifiée 224×224, sans le cadre
       + 2 recadrages de secours (centre, plus large) si la détection est incertaine
 → [3] embedding 256-d normalisé L2      (ViT-S/14 int8, ~25–40 ms CPU milieu de gamme)
 → [4] recherche brute dans le musée courant → top-5   (< 5 ms)
 → [5] reranking XFeat + RANSAC sur le top-5           (~40–80 ms, seulement quand c'est utile)
 → [6] score de confiance + vote temporel sur 3–5 images
 → [7] décision
```

**[1] Détection.** On entraîne un détecteur de quadrilatère (4 heatmaps de coins + score « tableau présent »). L'astuce, c'est que **les données d'entraînement sont gratuites** : on colle les images de référence en perspective aléatoire sur des photos de murs de musée, avec cadres synthétiques, reflets, flou, bruit de capteur et silhouettes en occlusion. Des centaines de milliers d'exemples sans annoter. Plan B immédiat : `VNDetectRectanglesRequest` sur iOS ; sur Android, recadrage central + multi-crops (l'embedding fine-tuné tolère déjà beaucoup).

**[3] Embedding.** Modèles comparés :

| Modèle | Licence | Pour Clara |
|---|---|---|
| CLIP / MobileCLIP | MobileCLIP v1 : licence Apple ; **MobileCLIP2 : poids sous `apple-amlr`, usage recherche** | très rapide, mais appris pour la *sémantique* image-texte : il confond les œuvres proches. Licence incompatible avec une app publiée (MobileCLIP2) |
| SigLIP 2 (B/16) | Apache 2.0 | bon généraliste, encore sémantique, plus lourd |
| **DINOv2 ViT-S/14** | Apache 2.0 | **meilleur point de départ** pour la recherche d'instance ; 21 M paramètres |
| MobileNetV3 / EfficientNet-Lite + GeM | Apache 2.0 | élève distillé si ViT-S est trop lent sur les petits téléphones |

Recommandation :
1. **Base** : DINOv2 ViT-S/14, sortie projetée en 256-d.
2. **Fine-tuning en metric learning** (ArcFace ou InfoNCE), où *chaque tableau est une classe*, sur le même générateur synthétique que le détecteur. Le modèle apprend « même tableau malgré reflet, perspective et occlusion » et « tableaux différents même si même artiste ». Ajouter des **négatifs difficiles** : les autres toiles de la même série et du même peintre.
3. **Généralisation** : le modèle doit fonctionner sur des tableaux *jamais vus à l'entraînement* (nouveaux musées sans réentraîner). On entraîne sur un grand ensemble (Commons, ~200 000 peintures du domaine public) et on évalue sur des musées tenus à l'écart.
4. **Export** ONNX, quantification int8 statique (calibration sur photos réelles), test d'écart fp32/int8.
5. Si besoin, **distillation** vers un MobileNetV3 pour les téléphones d'entrée de gamme.

**[5] Reranking géométrique.** Pour chaque candidat du top-5 : correspondances XFeat entre la photo rectifiée et les keypoints de référence, puis RANSAC homographie. Le **nombre d'inliers** est un signal très fiable : deux Nymphéas différents ont une couleur proche (embedding proche) mais aucune géométrie commune. On ne lance le reranking que si l'embedding hésite (marge top1-top2 faible), pour économiser la batterie.

**[6] Confiance.** On calibre un petit modèle logistique sur un jeu de test réel :

```
features = [cos_top1, cos_top1 - cos_top2, inliers_top1, inliers_top1 - inliers_top2,
            score_détection, accord_temporel (k/n images d'accord)]
p = sigmoid(w · features + b)
```

**[7] Décision** (seuils réglés pour viser < 1 % de fausses acceptations) :

| p | Interface |
|---|---|
| ≥ 0,90 | ajout direct, animation « nouvelle œuvre » (ou « déjà dans ta collection ») |
| 0,50 – 0,90 | « C'est bien celle-ci ? » + 2 alternatives en vignettes |
| < 0,50 | « Œuvre non reconnue » → proposer « Chercher partout » ou la recherche texte |

Une confirmation de l'utilisateur ajoute l'embedding de la photo rectifiée dans `user_embedding`, dans **sa** base locale : Clara reconnaît mieux, sur son appareil, les œuvres qu'il revoit.

### 4.3 Mode visée continue

Plutôt que « prendre une photo puis attendre », la caméra analyse en continu (~5 i/s). Dès que 3 images sur 5 donnent la même œuvre avec un score suffisant, Clara valide avec un retour haptique. C'est ce qui rend l'expérience instantanée, et le vote temporel élimine la plupart des erreurs dues aux reflets (un reflet bouge, le tableau non).

---

## 5. Recherche vectorielle locale

Mesures du banc d'essai [`tools/bench_search.py`](../tools/bench_search.py) (recherche brute exacte, 512-d, 3 vues par œuvre, CPU de bureau, numpy) :

| Œuvres | Vecteurs | f32 | int8 | Temps f32 |
|---|---|---|---|---|
| 1 500 (un musée) | 4 500 | 9 Mo | 2,3 Mo | 0,4 ms |
| 10 000 (Louvre, peintures + arts graphiques exposés) | 30 000 | 61 Mo | 15 Mo | 3 ms |
| 50 000 | 150 000 | 307 Mo | 77 Mo | 22 ms |
| 300 000 (« partout ») | 900 000 | 1,8 Go | 461 Mo | 148 ms |

En 256-d (ce que je recommande), diviser les tailles par deux. Sur téléphone, compter 3 à 6 fois plus lent qu'ici en f32. En natif, l'int8 avec `dotprod` NEON est 2 à 4 fois **plus rapide** que le f32 (le banc numpy montre l'inverse parce que numpy n'a pas de produit int8 optimisé).

**Décision :**
- **Musée courant : brute force int8**, vecteurs mmappés depuis `vectors.bin`. Exact, sans index à construire, quelques Mo de RAM, < 5 ms. FAISS, HNSW ou SQLite vectoriel n'apportent rien à cette taille.
- **« Reconnaître partout » : USearch (HNSW, i8, mmap)** construit à l'avance dans le pipeline. 300 000 œuvres × 1 vue × 256-d ≈ 77 Mo + graphe, < 10 ms par requête, avec seulement les pages utiles chargées en RAM. Filtrage par pays ou ville en option.
- **Écartés** : FAISS (excellent mais lourd à compiler pour mobile, ses avantages IVF-PQ ne servent qu'au-delà du million de vecteurs) ; `sqlite-vec` (correct, mais c'est de la brute force qu'on écrit en 30 lignes avec plus de contrôle) ; l'extension `sqlite-vector` de sqliteai (vérifier sa licence, qui ne me semble pas être une licence libre OSI).

Les vecteurs restent **hors de SQLite** : un fichier plat aligné se mmappe et se parcourt en SIMD sans désérialisation. SQLite ne garde que la correspondance index → œuvre.

---

## 6. Contexte musée

Détection du musée, sans réseau :
1. **GPS** (fonctionne en mode avion) → géofence sur les coordonnées des musées installés (`museum.geofence_m`).
2. Sinon, **dernier musée utilisé** dans les 6 dernières heures.
3. Sinon, **choix manuel** dans la liste des packs installés.
4. **Correction automatique** : si plusieurs scans de suite échouent dans le musée courant mais réussissent nettement dans un autre pack installé, proposer « Tu es au Musée X ? ».

Recherche :
```
photo → musée courant (brute force, top-5)
      → si p < 0,5 : musées installés de la même ville
      → si toujours rien, ou mode « Reconnaître partout » : index global USearch
```

Le contexte améliore la précision (moins de tableaux qui se ressemblent), la vitesse et la batterie : on compare à 1 500 tableaux au lieu de 300 000.

---

## 7. Fonctionnalités et données utilisateur

Toutes hors ligne, dans `user.db` :
- **Ma collection** : grille par musée, tri par date de découverte ou par artiste.
- **Progression** : `œuvres collectionnées comptables / number_of_works`, par musée, par collection et, au niveau 3, par salle.
- **Fiches** : métadonnées du pack, vignette si les droits le permettent, sinon la photo de l'utilisateur.
- **Recherche** : FTS5 sur titre, artiste, description, insensible aux accents.
- **Favoris, notes, historique.**
- **Ajout manuel** depuis la recherche (`via = 'manual'`), distingué des scans dans les statistiques.
- **Export/import** de `user.db` (fichier local) pour changer de téléphone sans compte ni serveur.
- Les photos de scan sont **facultatives** : par défaut on ne garde que la vignette rectifiée, purgeable à tout moment.

---

## 8. Pipeline de build (hors appareil)

Python, sur ton PC, lancé à la main ou par une CI :

```
ingest/      wikidata.py (SPARQL), europeana.py, api_<musee>.py
merge/       rapprochement par n° d'inventaire + QID, dédoublonnage, priorité des sources
rights/      classification open/restricted/unknown, calcul domaine public (artiste mort + 70 ans)
status/      statut d'exposition par musée, niveau de données
images/      téléchargement, rognage du cadre, vignettes webp
embed/       vues de référence (complète + recadrages), embeddings, keypoints XFeat
qa/          doublons visuels (paires d'œuvres trop proches → hard negatives / alerte),
             images manquantes, taux d'œuvres sans image
pack/        catalog.db, vectors.bin, keypoints.bin, manifest.json, zip + sha256
```

Le contrôle qualité visuel est important : deux œuvres différentes dont les embeddings de référence sont presque identiques sont soit un doublon de données, soit une paire difficile à traiter explicitement.

---

## 9. Évaluation : la seule mesure qui compte

Un **jeu de test réel** : 300 à 500 photos prises *dans* 2 ou 3 musées, au téléphone, dans de vraies conditions (biais, reflets, visiteurs, cartels, sculptures hors base). Indicateurs :

| Indicateur | Objectif v1 |
|---|---|
| Top-1 en mode musée (peintures présentes dans la base) | ≥ 95 % |
| Fausses acceptations au seuil automatique | < 1 % |
| Rejet correct des objets hors base | ≥ 90 % |
| Latence de la visée à la validation (p95, milieu de gamme) | < 1,5 s |
| Taille de l'app (sans pack) | < 60 Mo |

Musées pilotes conseillés :
- **SMK** (Copenhague) : données CC0, et l'API indique la salle où l'œuvre est accrochée (si ce champ manque, l'œuvre n'est pas exposée). Niveau 3 directement.
- **Rijksmuseum** : données et images CC0, API riche. Le statut d'exposition reste à vérifier dans l'API.
- **Belvedere** (Vienne) : politique Open Content pour les œuvres libres de droits (téléchargement avec inscription) ; pas d'API d'exposition connue, donc niveau 0–1.

Orsay et le Louvre viendront ensuite, au niveau 0–1.

---

## 10. Feuille de route

| Phase | Contenu | Critère de sortie |
|---|---|---|
| **P0 — Preuve** (Python, [`pipeline/`](../pipeline/README.md) : code prêt, mesures réelles à lancer) | pipeline Wikidata → 1 musée ; comparaison DINOv2 / SigLIP 2 / MobileCLIP sur le jeu de test réel, avec et sans reranking XFeat | chiffres top-1 / fausses acceptations, choix du modèle |
| **P1 — MVP Android** | Flutter + `clara_core` ; 1 pack ; visée continue ; collection ; progression ; fiches ; tout en mode avion | parcours complet dans un vrai musée sans réseau |
| **P2 — Robustesse** | détecteur 4 coins synthétique ; fine-tuning metric learning ; calibration de la confiance ; vues confirmées par l'utilisateur (`user_embedding`) | objectifs du §9 atteints |
| **P3 — Échelle** | 10+ musées, niveaux de données, mises à jour de packs, « Reconnaître partout » (USearch), iOS (Core ML EP) | 2 OS, index global < 100 Mo |

---

## 11. Risques

| Risque | Parade |
|---|---|
| Statut d'exposition faux | niveaux de données explicites, date de vérification affichée, scans de l'utilisateur comme preuve locale |
| Séries et répliques confondues | hard negatives au fine-tuning + reranking géométrique + confirmation à 2 alternatives |
| Droits sur les images | `rights_class` par image, fiches sans image pour les œuvres protégées, avis juridique avant publication |
| Œuvres d'expositions temporaires | rejet propre (« non reconnue ») ; packs « expo » possibles plus tard |
| Batterie en visée continue | 5 i/s plafonné, arrêt automatique après 20 s sans tableau, reranking seulement en cas d'hésitation |
| Licences des modèles | uniquement des poids Apache 2.0 / MIT (DINOv2, XFeat, MobileNetV3) ; MobileCLIP2 exclu (licence recherche) |
