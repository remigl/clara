# Clara

Scanner un tableau dans un musée, le reconnaître **sur l'appareil**, l'ajouter à sa collection et suivre sa progression dans le musée.

- 100 % hors ligne une fois le pack du musée téléchargé : aucune photo ne quitte le téléphone.
- Données ouvertes uniquement : Wikidata, Europeana, open data des musées.
- Reconnaissance par computer vision locale : détection du tableau, embedding, recherche vectorielle, vérification géométrique.

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | architecture complète : données, reconnaissance, recherche vectorielle, contexte musée, feuille de route |
| [`data/schema.sql`](data/schema.sql) | schéma SQLite (`catalog.db` par pack, `user.db` pour l'utilisateur) |
| [`pipeline/`](pipeline/README.md) | phase P0 en Python : reconnaissance, évaluation Met Dataset, construction des packs (SMK, Wikidata) |
| [`tools/bench_search.py`](tools/bench_search.py) | banc d'essai de la recherche vectorielle brute (numpy) |

## État

Phase P0 : le code est prêt et testé hors ligne ; les mesures sur données réelles se lancent depuis `pipeline/` (voir son README). Feuille de route : `docs/ARCHITECTURE.md` §10.
