import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';

import 'core.dart';
import 'screens.dart' show artThumb, showAdded;

// ---------------------------------------------------------------------------- navigation

/// Ouvre la fiche d'une œuvre. Avec [list], on passe d'une œuvre à l'autre par balayage horizontal,
/// dans l'ordre de la liste affichée.
void openArtwork(BuildContext context, AppState s, Artwork a, {List<Artwork>? list}) {
  final items = (list == null || list.isEmpty) ? [a] : list;
  final i = items.indexWhere((x) => x.id == a.id);
  Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => ArtworkPager(state: s, items: i < 0 ? [a] : items, initial: i < 0 ? 0 : i)));
}

class ArtworkPager extends StatefulWidget {
  const ArtworkPager({super.key, required this.state, required this.items, required this.initial});
  final AppState state;
  final List<Artwork> items;
  final int initial;

  @override
  State<ArtworkPager> createState() => _ArtworkPagerState();
}

class _ArtworkPagerState extends State<ArtworkPager> {
  late final PageController _pc = PageController(initialPage: widget.initial);
  late int _index = widget.initial;

  @override
  void dispose() {
    _pc.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    final items = widget.items;
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) {
        final a = items[_index];
        final fav = s.isFavorite(a.id);
        return Scaffold(
          appBar: AppBar(
            title: Text(a.pack.name, maxLines: 1, overflow: TextOverflow.ellipsis),
            bottom: items.length > 1
                ? PreferredSize(
                    preferredSize: const Size.fromHeight(18),
                    child: Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Text('${_index + 1} / ${items.length}  ·  balaie pour passer à la suivante',
                          style: Theme.of(context).textTheme.labelSmall),
                    ),
                  )
                : null,
            actions: [
              IconButton(
                tooltip: fav ? 'Retirer des favoris' : 'Ajouter aux favoris',
                icon: Icon(fav ? Icons.favorite : Icons.favorite_border, color: fav ? Colors.redAccent : null),
                onPressed: () => s.toggleFavorite(a.id),
              ),
            ],
          ),
          body: PageView.builder(
            controller: _pc,
            itemCount: items.length,
            onPageChanged: (i) => setState(() => _index = i),
            itemBuilder: (context, i) => ArtworkBody(key: ValueKey(items[i].id), state: s, artwork: items[i]),
          ),
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------- fiche

class ArtworkBody extends StatefulWidget {
  const ArtworkBody({super.key, required this.state, required this.artwork});
  final AppState state;
  final Artwork artwork;

  @override
  State<ArtworkBody> createState() => _ArtworkBodyState();
}

class _ArtworkBodyState extends State<ArtworkBody> {
  bool _artistTab = false;
  int _photo = 0; // index dans [photos personnelles…, image du musée]

  static const _sectionIcons = {
    'Description': Icons.subject,
    'À observer': Icons.visibility_outlined,
    'Contexte et histoire': Icons.history_edu_outlined,
    'Anecdotes': Icons.lightbulb_outline,
    'Iconographie': Icons.auto_stories_outlined,
    'Création': Icons.brush_outlined,
    'Provenance': Icons.account_balance_outlined,
    'Expositions': Icons.event_outlined,
  };

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    final a = widget.artwork;
    final theme = Theme.of(context);
    final artist = a.pack.artist(a.artistId);
    final photos = s.photosOf(a.id);
    final views = <Widget Function(BoxFit)>[
      for (final (_, f) in photos) (fit) => Image.file(f, fit: fit, errorBuilder: (_, __, ___) => artThumb(a, fit: fit)),
      (fit) => artThumb(a, fit: fit),
    ];
    final cur = _photo.clamp(0, views.length - 1);

    return ListView(padding: const EdgeInsets.fromLTRB(16, 12, 16, 40), children: [
      // --- image : photo personnelle si elle existe, sinon l'image du musée
      GestureDetector(
        onTap: () => Navigator.of(context).push(MaterialPageRoute(
            builder: (_) => Scaffold(
                  backgroundColor: Colors.black,
                  appBar: AppBar(backgroundColor: Colors.black, foregroundColor: Colors.white),
                  body: InteractiveViewer(maxScale: 6, child: Center(child: views[cur](BoxFit.contain))),
                ))),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(14),
          child: Container(
            height: 340,
            color: const Color(0xFFF3EEE8),
            child: views[cur](BoxFit.contain),
          ),
        ),
      ),
      if (photos.isNotEmpty) ...[
        const SizedBox(height: 8),
        _PhotoStrip(
          state: s,
          artwork: a,
          photos: photos,
          selected: cur,
          onSelect: (i) => setState(() => _photo = i),
        ),
      ],
      const SizedBox(height: 14),
      Text(a.title, style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600)),
      const SizedBox(height: 2),
      Text([a.artist ?? 'Artiste inconnu', if (a.date != null) a.date!].join(' · '),
          style: theme.textTheme.titleMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
      const SizedBox(height: 12),

      // --- note personnelle, juste sous l'image
      NoteBox(key: ValueKey('note-${a.id}'), state: s, artworkId: a.id),
      const SizedBox(height: 8),
      _CollectRow(state: s, artwork: a),
      const SizedBox(height: 16),

      // --- deux niveaux : l'œuvre / l'artiste
      SegmentedButton<bool>(
        segments: const [
          ButtonSegment(value: false, icon: Icon(Icons.image_outlined), label: Text('L\'œuvre')),
          ButtonSegment(value: true, icon: Icon(Icons.person_outline), label: Text('L\'artiste')),
        ],
        selected: {_artistTab},
        showSelectedIcon: false,
        onSelectionChanged: (v) => setState(() => _artistTab = v.first),
      ),
      const SizedBox(height: 4),
      ...(_artistTab ? _artistPart(context, a, artist) : _artworkPart(context, a)),
    ]);
  }

  // ------------------------------------------------------------------ niveau 1 : l'œuvre

  List<Widget> _artworkPart(BuildContext context, Artwork a) {
    final theme = Theme.of(context);
    String? dims;
    if (a.heightCm != null && a.widthCm != null) {
      String f(double v) => v == v.roundToDouble() ? v.toStringAsFixed(0) : v.toStringAsFixed(1);
      dims = '${f(a.heightCm!)} × ${f(a.widthCm!)} cm';
    }
    final facts = <(String, String?)>[
      ('Date', a.date),
      ('Technique', a.medium),
      ('Dimensions', dims),
      ...a.facts.where((f) => f.$1 != 'Matériaux' || a.medium == null).map((f) => (f.$1, f.$2)),
      ('Musée', '${a.pack.name}, ${a.pack.city}'),
      ('Emplacement', a.place),
      ('Inventaire', a.inventory),
    ];
    final glossary = a.pack.glossaryFor(a.id);
    return [
      if (a.description != null && a.description!.isNotEmpty) ...[
        _heading(context, 'À propos', Icons.info_outline),
        ExpandableText(a.description!, style: theme.textTheme.bodyLarge?.copyWith(height: 1.5)),
        _credit(context, a.descriptionCredit),
      ],
      _heading(context, 'En bref', Icons.list_alt_outlined),
      for (final (label, value) in facts)
        if (value != null && value.isNotEmpty) _row(context, label, value),
      for (final (title, items) in a.sections)
        if (items.isNotEmpty) ...[
          _heading(context, title, _sectionIcons[title] ?? Icons.notes_outlined),
          for (final (label, value) in items)
            if (label == 'Source')
              _credit(context, value)
            else if (label.isEmpty && value.length > 300)
              ExpandableText(value, style: theme.textTheme.bodyMedium?.copyWith(height: 1.5))
            else
              _row(context, label, value),
        ],
      if (glossary.isNotEmpty) ...[
        _heading(context, 'Pour comprendre', Icons.school_outlined),
        for (final g in glossary)
          Card(
            margin: const EdgeInsets.only(bottom: 8),
            elevation: 0,
            color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
            child: ExpansionTile(
              shape: const Border(),
              title: Text(g.term[0].toUpperCase() + g.term.substring(1)),
              childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              expandedCrossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(g.text, style: theme.textTheme.bodyMedium?.copyWith(height: 1.45)),
                _credit(context, g.credit),
              ],
            ),
          ),
      ],
      _heading(context, 'Sources', Icons.link),
      if (a.credit != null) Text(a.credit!, style: theme.textTheme.bodySmall),
      if (a.sourceUrl != null) SelectableText(a.sourceUrl!, style: theme.textTheme.bodySmall),
    ];
  }

  // ------------------------------------------------------------------ niveau 2 : l'artiste

  List<Widget> _artistPart(BuildContext context, Artwork a, ArtistInfo? artist) {
    final theme = Theme.of(context);
    final s = widget.state;
    final others = a.pack.otherWorksBy(a.artistId, a.id);
    if (artist == null) {
      return [
        const SizedBox(height: 24),
        Text(a.artist ?? 'Artiste inconnu', style: theme.textTheme.titleLarge),
        const SizedBox(height: 8),
        const Text('Pas d\'informations publiques sur cet artiste pour l\'instant.'),
      ];
    }
    final life = [
      if (artist.birthYear != null || artist.deathYear != null) '${artist.birthYear ?? '?'} – ${artist.deathYear ?? ''}',
      if (artist.nationality != null) artist.nationality!,
    ].join(' · ');
    return [
      const SizedBox(height: 20),
      Text(artist.name, style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600)),
      if (life.isNotEmpty) Text(life, style: theme.textTheme.titleSmall),
      if (artist.lifeLine != null)
        Padding(padding: const EdgeInsets.only(top: 2), child: Text(artist.lifeLine!, style: theme.textTheme.bodySmall)),
      if (artist.bio != null) ...[
        _heading(context, 'Biographie', Icons.person_outline),
        ExpandableText(artist.bio!, style: theme.textTheme.bodyLarge?.copyWith(height: 1.5)),
        _credit(context, artist.bioCredit),
      ],
      if (artist.movements != null || artist.occupations != null || artist.style != null) ...[
        _heading(context, 'Style et mouvements', Icons.palette_outlined),
        if (artist.movements != null) _row(context, 'Mouvements', artist.movements!),
        if (artist.occupations != null) _row(context, 'Métier', artist.occupations!),
        if (artist.teachers != null) _row(context, 'Maîtres', artist.teachers!),
        if (artist.students != null) _row(context, 'Élèves', artist.students!),
        if (artist.style != null) ...[
          const SizedBox(height: 6),
          ExpandableText(artist.style!, style: theme.textTheme.bodyMedium?.copyWith(height: 1.5)),
        ],
      ],
      if (artist.notableWorks != null) ...[
        _heading(context, 'Œuvres principales', Icons.star_outline),
        Text(artist.notableWorks!, style: theme.textTheme.bodyMedium?.copyWith(height: 1.5)),
      ],
      if (others.isNotEmpty) ...[
        _heading(context, 'Du même artiste, ici', Icons.collections_outlined),
        SizedBox(
          height: 160,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            itemCount: others.length,
            separatorBuilder: (_, __) => const SizedBox(width: 10),
            itemBuilder: (context, i) {
              final o = others[i];
              return InkWell(
                onTap: () => openArtwork(context, s, o, list: others),
                child: SizedBox(
                  width: 112,
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Expanded(
                      child: Stack(fit: StackFit.expand, children: [
                        ClipRRect(borderRadius: BorderRadius.circular(8), child: artThumb(o)),
                        if (s.isCollected(o.id))
                          const Positioned(right: 4, top: 4, child: Icon(Icons.check_circle, color: Colors.white, size: 18)),
                      ]),
                    ),
                    const SizedBox(height: 4),
                    Text(o.title, maxLines: 2, overflow: TextOverflow.ellipsis, style: theme.textTheme.bodySmall),
                  ]),
                ),
              );
            },
          ),
        ),
      ],
      if (artist.anecdotes != null) ...[
        _heading(context, 'Anecdotes', Icons.lightbulb_outline),
        ExpandableText(artist.anecdotes!, style: theme.textTheme.bodyMedium?.copyWith(height: 1.5)),
      ],
      _credit(context, artist.extraCredit),
    ];
  }

  // ------------------------------------------------------------------ petits éléments

  Widget _heading(BuildContext context, String t, IconData icon) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(top: 26, bottom: 8),
      child: Row(children: [
        Icon(icon, size: 20, color: theme.colorScheme.primary),
        const SizedBox(width: 8),
        Expanded(child: Text(t, style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600))),
      ]),
    );
  }

  Widget _row(BuildContext context, String label, String value) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: label.isEmpty
          ? Text(value, style: theme.textTheme.bodyMedium?.copyWith(height: 1.45))
          : Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              SizedBox(
                  width: 116,
                  child: Text(label, style: theme.textTheme.labelLarge?.copyWith(color: theme.colorScheme.onSurfaceVariant))),
              Expanded(child: Text(value)),
            ]),
    );
  }

  Widget _credit(BuildContext context, String? c) => c == null
      ? const SizedBox.shrink()
      : Padding(
          padding: const EdgeInsets.only(top: 6),
          child: Text(c, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.black45)));
}

/// Texte long replié sur quelques lignes, « Lire la suite » pour tout afficher.
class ExpandableText extends StatefulWidget {
  const ExpandableText(this.text, {super.key, this.style, this.lines = 6});
  final String text;
  final TextStyle? style;
  final int lines;

  @override
  State<ExpandableText> createState() => _ExpandableTextState();
}

class _ExpandableTextState extends State<ExpandableText> {
  bool _open = false;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, c) {
      final tp = TextPainter(
        text: TextSpan(text: widget.text, style: widget.style ?? DefaultTextStyle.of(context).style),
        maxLines: widget.lines,
        textDirection: TextDirection.ltr,
        textScaler: MediaQuery.textScalerOf(context),
      )..layout(maxWidth: c.maxWidth);
      final long = tp.didExceedMaxLines;
      return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        AnimatedSize(
          duration: const Duration(milliseconds: 200),
          alignment: Alignment.topCenter,
          child: Text(widget.text,
              style: widget.style,
              maxLines: _open || !long ? null : widget.lines,
              overflow: _open || !long ? null : TextOverflow.fade),
        ),
        if (long)
          TextButton(
            style: TextButton.styleFrom(padding: EdgeInsets.zero, visualDensity: VisualDensity.compact),
            onPressed: () => setState(() => _open = !_open),
            child: Text(_open ? 'Réduire' : 'Lire la suite'),
          ),
      ]);
    });
  }
}

class _CollectRow extends StatelessWidget {
  const _CollectRow({required this.state, required this.artwork});
  final AppState state;
  final Artwork artwork;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (state.isCollected(artwork.id)) {
      return Row(children: [
        Icon(Icons.check_circle, color: theme.colorScheme.primary, size: 20),
        const SizedBox(width: 8),
        const Text('Dans ta collection'),
      ]);
    }
    return Align(
      alignment: Alignment.centerLeft,
      child: OutlinedButton.icon(
        icon: const Icon(Icons.add),
        label: const Text('Je l\'ai vue : ajouter à ma collection'),
        onPressed: () => showAdded(context, state, artwork, state.add(artwork, via: 'manual')),
      ),
    );
  }
}

/// Vignettes des photos personnelles (et de l'image du musée, en dernier) ; appui long pour supprimer.
class _PhotoStrip extends StatelessWidget {
  const _PhotoStrip(
      {required this.state, required this.artwork, required this.photos, required this.selected, required this.onSelect});
  final AppState state;
  final Artwork artwork;
  final List<(int, File)> photos;
  final int selected;
  final ValueChanged<int> onSelect;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    Widget tile(int i, Widget child, String label) => GestureDetector(
          onTap: () => onSelect(i),
          onLongPress: i < photos.length
              ? () async {
                  final ok = await showDialog<bool>(
                    context: context,
                    builder: (c) => AlertDialog(
                      title: const Text('Supprimer cette photo ?'),
                      actions: [
                        TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Annuler')),
                        FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Supprimer')),
                      ],
                    ),
                  );
                  if (ok == true) {
                    state.deletePhoto(photos[i].$1);
                    onSelect(0);
                  }
                }
              : null,
          child: Tooltip(
            message: label,
            child: Container(
              width: 56,
              height: 56,
              margin: const EdgeInsets.only(right: 8),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                    color: i == selected ? theme.colorScheme.primary : Colors.transparent, width: 2),
              ),
              child: ClipRRect(borderRadius: BorderRadius.circular(6), child: child),
            ),
          ),
        );
    return SizedBox(
      height: 56,
      child: Row(children: [
        Expanded(
          child: ListView(scrollDirection: Axis.horizontal, children: [
            for (var i = 0; i < photos.length; i++)
              tile(i, Image.file(photos[i].$2, fit: BoxFit.cover, cacheWidth: 160), 'Ta photo'),
            tile(photos.length, artThumb(artwork), 'Image du musée'),
          ]),
        ),
        Text(selected < photos.length ? 'Ta photo' : 'Image du musée', style: theme.textTheme.labelSmall),
      ]),
    );
  }
}

// ---------------------------------------------------------------------------- note personnelle

/// Note libre, enregistrée automatiquement pendant la saisie.
class NoteBox extends StatefulWidget {
  const NoteBox({super.key, required this.state, required this.artworkId});
  final AppState state;
  final String artworkId;

  @override
  State<NoteBox> createState() => _NoteBoxState();
}

class _NoteBoxState extends State<NoteBox> {
  late final TextEditingController _c = TextEditingController(text: widget.state.noteOf(widget.artworkId) ?? '');
  final _focus = FocusNode();
  bool _editing = false;
  Timer? _debounce;

  void _save() {
    _debounce?.cancel();
    widget.state.saveNote(widget.artworkId, _c.text);
  }

  @override
  void dispose() {
    _save();
    _debounce?.cancel();
    _c.dispose();
    _focus.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final has = _c.text.trim().isNotEmpty;
    if (!_editing && !has) {
      return Align(
        alignment: Alignment.centerLeft,
        child: TextButton.icon(
          style: TextButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 4)),
          icon: const Icon(Icons.edit_note),
          label: const Text('Ajouter une note'),
          onPressed: () {
            setState(() => _editing = true);
          },
        ),
      );
    }
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 8, 4, 8),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF8E7),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFEADBB8)),
      ),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Padding(padding: EdgeInsets.only(top: 2), child: Icon(Icons.sticky_note_2_outlined, size: 20, color: Color(0xFF9C7A3C))),
        const SizedBox(width: 8),
        Expanded(
          child: _editing
              ? TextField(
                  controller: _c,
                  focusNode: _focus,
                  autofocus: true,
                  maxLines: null,
                  minLines: 2,
                  textCapitalization: TextCapitalization.sentences,
                  decoration: const InputDecoration.collapsed(hintText: 'Ce que tu as ressenti, remarqué…'),
                  onChanged: (_) {
                    _debounce?.cancel();
                    _debounce = Timer(const Duration(milliseconds: 500), _save);
                  },
                  onTapOutside: (_) => _focus.unfocus(),
                )
              : GestureDetector(
                  onTap: () {
                    setState(() => _editing = true);
                  },
                  child: Text(_c.text, style: theme.textTheme.bodyMedium?.copyWith(height: 1.4)),
                ),
        ),
        if (_editing)
          IconButton(
            tooltip: 'Terminer',
            visualDensity: VisualDensity.compact,
            icon: const Icon(Icons.check),
            onPressed: () {
              _save();
              _focus.unfocus();
              setState(() => _editing = false);
            },
          )
        else
          PopupMenuButton<String>(
            icon: const Icon(Icons.more_vert, size: 20),
            onSelected: (v) {
              if (v == 'edit') {
                setState(() => _editing = true);
              } else {
                _c.clear();
                _save();
                setState(() {});
              }
            },
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'edit', child: Text('Modifier')),
              PopupMenuItem(value: 'delete', child: Text('Supprimer la note')),
            ],
          ),
      ]),
    );
  }
}
