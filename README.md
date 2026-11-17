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
| [`tools/bench_search.py`](tools/bench_search.py) | banc d'essai de la recherche vectorielle brute (numpy) |

## État

Phase P0 (conception et preuve de concept). Voir la feuille de route dans `docs/ARCHITECTURE.md` §10.
