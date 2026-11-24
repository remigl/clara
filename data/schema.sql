-- Clara — schéma SQLite embarqué
-- Deux bases physiquement séparées :
--   catalog.db  : données de référence, en lecture seule, remplacées par packs (une par musée + un index global)
--   user.db     : données de l'utilisateur, jamais écrasées par une mise à jour du catalogue
-- Les identifiants d'œuvres sont stables (voir ARCHITECTURE.md §4.2), ce qui permet
-- de mettre à jour le catalogue sans casser la collection de l'utilisateur.

----------------------------------------------------------------------
-- catalog.db
----------------------------------------------------------------------
PRAGMA foreign_keys = ON;

CREATE TABLE pack (                      -- un pack = un musée téléchargé
  id               TEXT PRIMARY KEY,     -- 'fr-orsay'
  version          INTEGER NOT NULL,     -- incrémenté à chaque build
  built_at         TEXT NOT NULL,        -- ISO-8601
  model_id         TEXT NOT NULL,        -- ex. 'clara-embed-v1' : les embeddings sont liés au modèle
  embedding_dim    INTEGER NOT NULL,
  embedding_dtype  TEXT NOT NULL,        -- 'int8' | 'f16'
  sha256           TEXT NOT NULL
);

CREATE TABLE museum (
  id               TEXT PRIMARY KEY,     -- 'fr-orsay'
  name             TEXT NOT NULL,
  name_local       TEXT,
  city             TEXT NOT NULL,
  country          TEXT NOT NULL,        -- ISO 3166-1 alpha-2
  lat              REAL NOT NULL,
  lon              REAL NOT NULL,
  geofence_m       INTEGER DEFAULT 250,  -- rayon de détection automatique du musée
  website          TEXT,
  image_path       TEXT,                 -- fichier local dans le pack
  wikidata_qid     TEXT,                 -- 'Q23402'
  data_quality     INTEGER NOT NULL DEFAULT 0, -- niveau 0..4, voir ARCHITECTURE.md §3.3
  number_of_works  INTEGER NOT NULL,     -- dénominateur de la progression (œuvres "comptables")
  pack_id          TEXT REFERENCES pack(id)
);

CREATE TABLE collection (                -- département / ensemble cohérent ("Peintures", "Impressionnisme")
  id               TEXT PRIMARY KEY,
  museum_id        TEXT NOT NULL REFERENCES museum(id),
  name             TEXT NOT NULL,
  number_of_works  INTEGER NOT NULL
);

CREATE TABLE source (                    -- provenance des données
  id               TEXT PRIMARY KEY,     -- 'europeana', 'wikidata', 'smk-api', 'parismusees-opendata'
  name             TEXT NOT NULL,
  url              TEXT,
  license          TEXT NOT NULL         -- licence des MÉTADONNÉES (ex. CC0)
);

CREATE TABLE artist (
  id               TEXT PRIMARY KEY,     -- QID Wikidata si possible
  name             TEXT NOT NULL,
  birth_year       INTEGER,
  death_year       INTEGER,              -- sert au calcul domaine public (mort + 70 ans)
  nationality      TEXT,
  birth_place      TEXT,
  death_place      TEXT,
  occupations      TEXT,                 -- « peintre, graveur »
  movements        TEXT,
  teachers         TEXT,                 -- maîtres (P1066)
  students         TEXT,                 -- élèves (P802)
  bio              TEXT,                 -- introduction Wikipédia
  bio_credit       TEXT,
  notable_works    TEXT,                 -- principales œuvres (P800)
  style            TEXT,                 -- section « style / œuvre » de l'article
  anecdotes        TEXT,                 -- vie, postérité, anecdotes
  extra_credit     TEXT
);

CREATE TABLE glossary (                  -- « pour comprendre » : mouvements, genres expliqués
  term             TEXT PRIMARY KEY,
  text             TEXT NOT NULL,
  credit           TEXT
);

CREATE TABLE artwork_glossary (
  artwork_id       TEXT NOT NULL,
  term             TEXT NOT NULL,
  PRIMARY KEY (artwork_id, term)
);

CREATE TABLE artwork (
  id               TEXT PRIMARY KEY,     -- 'fr-orsay:RF-1986-3' (musée + n° d'inventaire normalisé)
  museum_id        TEXT NOT NULL REFERENCES museum(id),
  collection_id    TEXT REFERENCES collection(id),
  title            TEXT NOT NULL,
  title_local      TEXT,
  artist_id        TEXT REFERENCES artist(id),
  artist_display   TEXT,                 -- "Claude Monet", "Atelier de Rubens", "Anonyme"
  date_display     TEXT,                 -- "vers 1872"
  date_start       INTEGER,
  date_end         INTEGER,
  description      TEXT,                 -- texte de présentation (catalogue du musée, Wikipédia…)
  description_credit TEXT,               -- auteur / licence du texte (ex. « Wikipédia, CC BY-SA 4.0 »)
  artist_life      TEXT,                 -- « 1562–1638 »
  artist_nationality TEXT,
  facts_json       TEXT,                 -- détails affichés tels quels : [["Genre","portrait"], ["Mouvement","baroque"], …]
  sections_json    TEXT,                 -- [["Provenance", [["1650–1700","X"], …]], ["Iconographie", …]]
  inventory_number TEXT,
  medium           TEXT,
  height_cm        REAL,
  width_cm         REAL,
  object_type      TEXT NOT NULL DEFAULT 'painting', -- painting | drawing | print | sculpture | ...
  room             TEXT,
  floor            TEXT,
  exhibition_status TEXT NOT NULL DEFAULT 'unknown',
      -- on_display | not_on_display | temporarily_absent | in_storage | on_loan | unknown
      -- (not_on_display : la source dit seulement « pas exposée », sans préciser où)
  status_source_id TEXT REFERENCES source(id),
  status_checked_at TEXT,                -- date de dernière vérification du statut
  counts_for_progress INTEGER NOT NULL DEFAULT 1, -- inclus dans le dénominateur du musée ?
  wikidata_qid     TEXT,
  source_id        TEXT NOT NULL REFERENCES source(id),
  source_url       TEXT,
  updated_at       TEXT NOT NULL
);
CREATE INDEX artwork_museum ON artwork(museum_id);
CREATE INDEX artwork_status ON artwork(museum_id, exhibition_status);

CREATE TABLE artwork_image (             -- image AFFICHÉE dans la fiche (peut être absente)
  artwork_id       TEXT NOT NULL REFERENCES artwork(id),
  local_path       TEXT,                 -- vignette embarquée, NULL si droits insuffisants
  width            INTEGER,
  height           INTEGER,
  rights_uri       TEXT NOT NULL,        -- ex. http://creativecommons.org/publicdomain/mark/1.0/
  rights_class     TEXT NOT NULL,        -- open | restricted | permission | unknown
  credit_line      TEXT,                 -- attribution à afficher
  original_url     TEXT,
  PRIMARY KEY (artwork_id)
);

CREATE TABLE artwork_embedding (         -- une œuvre peut avoir plusieurs vues de référence
  row              INTEGER PRIMARY KEY,  -- ligne dans vectors.bin (int8[dim]) et scales.bin (f32)
  artwork_id       TEXT NOT NULL REFERENCES artwork(id),
  view_idx         INTEGER NOT NULL,     -- 0 = image complète, 1.. = recadrages / augmentations
  kind             TEXT NOT NULL,        -- 'full' | 'crop_center' | 'crop_tl' | 'augmented'
  UNIQUE (artwork_id, view_idx)
);
-- Les vecteurs restent hors de SQLite (fichier plat mmappé, parcouru en SIMD) : voir ARCHITECTURE.md §5.

CREATE TABLE artwork_keypoints (         -- descripteurs locaux pour la vérification géométrique (reranking)
  artwork_id       TEXT PRIMARY KEY REFERENCES artwork(id),
  offset           INTEGER NOT NULL,     -- position dans keypoints.bin
  length           INTEGER NOT NULL      -- P0 : SIFT (u32 n, f32 xy[n,2], u8 desc[n,128]) ; app : XFeat
);

-- recherche plein texte hors ligne (titre, artiste, description)
CREATE VIRTUAL TABLE artwork_fts USING fts5(
  title, artist_display, description, content='artwork', content_rowid='rowid',
  tokenize = 'unicode61 remove_diacritics 2'
);

----------------------------------------------------------------------
-- user.db
----------------------------------------------------------------------
CREATE TABLE user_embedding (            -- vues ajoutées après confirmation de l'utilisateur, sur SON appareil
  id               INTEGER PRIMARY KEY,
  artwork_id       TEXT NOT NULL,
  model_id         TEXT NOT NULL,
  vector           BLOB NOT NULL,        -- int8[dim] (petit volume : un BLOB suffit ici)
  scale            REAL NOT NULL,
  created_at       TEXT NOT NULL
);

CREATE TABLE scan (                      -- historique brut : chaque tentative de reconnaissance
  id               INTEGER PRIMARY KEY,
  created_at       TEXT NOT NULL,
  museum_id        TEXT,                 -- contexte au moment du scan (NULL = "Reconnaître partout")
  artwork_id       TEXT,                 -- NULL si aucune œuvre retenue
  confidence       REAL,
  decision         TEXT NOT NULL,        -- auto_accepted | user_confirmed | user_picked_alt | rejected | no_match
  candidates_json  TEXT,                 -- top-5 {id, score} pour diagnostic local
  photo_path       TEXT,                 -- facultatif, reste sur l'appareil, purgeable
  model_id         TEXT NOT NULL
);

CREATE TABLE collected (                 -- "Ma collection" : une ligne par œuvre découverte
  artwork_id       TEXT PRIMARY KEY,
  museum_id        TEXT NOT NULL,
  first_seen_at    TEXT NOT NULL,
  last_seen_at     TEXT NOT NULL,
  times_seen       INTEGER NOT NULL DEFAULT 1,
  via              TEXT NOT NULL         -- 'scan' | 'manual' (ajout manuel depuis la recherche)
);

CREATE TABLE viewed (                    -- fiches consultées sans scan
  artwork_id       TEXT PRIMARY KEY,
  last_viewed_at   TEXT NOT NULL
);

CREATE TABLE favorite (
  artwork_id       TEXT PRIMARY KEY,
  created_at       TEXT NOT NULL
);

CREATE TABLE note (
  id               INTEGER PRIMARY KEY,
  artwork_id       TEXT NOT NULL,
  body             TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL
);

CREATE TABLE id_alias (                  -- si un build renomme un id, on garde la trace
  old_id           TEXT PRIMARY KEY,
  new_id           TEXT NOT NULL
);

-- Progression d'un musée (calculée à la volée, ATTACH catalog.db AS c) :
-- SELECT COUNT(*) * 1.0 / m.number_of_works
-- FROM collected u JOIN c.artwork a ON a.id = u.artwork_id
-- JOIN c.museum m ON m.id = a.museum_id
-- WHERE a.museum_id = :museum AND a.counts_for_progress = 1;
