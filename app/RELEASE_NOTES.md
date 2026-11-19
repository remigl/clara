**v0.1.1** — corrige le plantage au démarrage de la v0.1.0 (« Could not find an implementation for ConvInteger ») : l'app embarque maintenant le modèle DINOv2 complet, celui des mesures P0. L'APK est plus lourd (~120 Mo).

Première version de Clara : reconnaître et collectionner les tableaux d'un musée, **100 % hors ligne**.

**Musée inclus** : SMK – Statens Museum for Kunst (Copenhague), peintures du domaine public actuellement exposées, avec leur salle.

**Ce que fait cette version**
- Scanner un tableau avec l'appareil photo : reconnaissance sur le téléphone (DINOv2, ONNX Runtime), aucune photo envoyée nulle part.
- « C'est bien celle-ci ? » : l'œuvre reconnue et deux alternatives ; tu confirmes, elle rejoint ta collection.
- Ma collection et progression dans le musée (œuvres découvertes / œuvres exposées).
- Liste et recherche des œuvres du musée, fiches, favoris, ajout manuel.
- Aucune permission Internet : l'app ne peut pas se connecter.

**Limites connues**
- Un seul musée, embarqué dans l'app.
- Toujours une confirmation : l'ajout automatique arrivera avec la vérification géométrique (mesurée en P0, pas encore portée sur le téléphone).
- Pas encore de visée continue : on appuie pour scanner.

Installation : télécharger `clara-v0.1.1.apk` sur le téléphone Android (8.0 ou plus récent, 64 bits) et l'ouvrir ; autoriser l'installation depuis cette source si Android le demande.
