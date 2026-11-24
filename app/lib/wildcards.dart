import 'package:flutter/material.dart';

import 'artwork_screen.dart';
import 'core.dart';
import 'screens.dart' show artThumb, showAdded;

/// Wildcards : des œuvres au hasard, dans des musées au hasard (parmi les musées téléchargés).
/// On balaie pour passer à la suivante ; les œuvres vues récemment ne reviennent pas.
class WildcardsScreen extends StatefulWidget {
  const WildcardsScreen({super.key, required this.state, required this.onOpenMuseums});
  final AppState state;
  final VoidCallback onOpenMuseums;

  @override
  State<WildcardsScreen> createState() => _WildcardsScreenState();
}

class _WildcardsScreenState extends State<WildcardsScreen> {
  final List<Artwork> _items = [];
  final PageController _pc = PageController(viewportFraction: 0.9);
  int _index = 0;

  @override
  void initState() {
    super.initState();
    _more();
  }

  @override
  void dispose() {
    _pc.dispose();
    super.dispose();
  }

  void _more() => _items.addAll(widget.state.wildcards(12));

  void _reshuffle() {
    setState(() {
      _items.clear();
      _more();
      _index = 0;
    });
    if (_pc.hasClients) _pc.jumpToPage(0);
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    final theme = Theme.of(context);
    return SafeArea(
      child: ListenableBuilder(
        listenable: s,
        builder: (context, _) {
          if (_items.isEmpty && s.packs.isNotEmpty) _more();
          return Column(children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 8, 4),
              child: Row(children: [
                Expanded(child: Text('Wildcards', style: theme.textTheme.headlineSmall)),
                IconButton(tooltip: 'Nouveau tirage', icon: const Icon(Icons.shuffle), onPressed: _reshuffle),
              ]),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Text(
                s.packs.length <= 1
                    ? 'Des œuvres au hasard. Télécharge d\'autres musées pour élargir le tirage.'
                    : 'Des œuvres au hasard parmi tes ${s.packs.length} musées. Balaie pour la suivante.',
                style: theme.textTheme.bodySmall,
              ),
            ),
            const SizedBox(height: 12),
            Expanded(
              child: _items.isEmpty
                  ? Center(
                      child: FilledButton.tonal(
                          onPressed: widget.onOpenMuseums, child: const Text('Télécharger des musées')))
                  : PageView.builder(
                      controller: _pc,
                      onPageChanged: (i) {
                        setState(() {
                          _index = i;
                          if (i >= _items.length - 3) _more();
                        });
                      },
                      itemCount: _items.length,
                      itemBuilder: (context, i) => _card(context, _items[i], i == _index),
                    ),
            ),
            const SizedBox(height: 16),
          ]);
        },
      ),
    );
  }

  Widget _card(BuildContext context, Artwork a, bool active) {
    final s = widget.state;
    final theme = Theme.of(context);
    final fav = s.isFavorite(a.id);
    final got = s.isCollected(a.id);
    return AnimatedScale(
      scale: active ? 1 : 0.94,
      duration: const Duration(milliseconds: 220),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 6),
        child: Card(
          clipBehavior: Clip.antiAlias,
          elevation: active ? 4 : 1,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          child: InkWell(
            onTap: () => openArtwork(context, s, a, list: _items),
            child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              Expanded(
                child: Container(
                  color: const Color(0xFF1E1A17),
                  child: artThumb(a, fit: BoxFit.contain),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 14, 16, 4),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(a.title,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 2),
                  Text([a.artist ?? 'Artiste inconnu', if (a.date != null) a.date!].join(' · '),
                      maxLines: 1, overflow: TextOverflow.ellipsis, style: theme.textTheme.bodyMedium),
                  const SizedBox(height: 2),
                  Text('${countryFlag(a.pack.country)}  ${a.pack.name}, ${a.pack.city}',
                      maxLines: 1, overflow: TextOverflow.ellipsis, style: theme.textTheme.bodySmall),
                ]),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(8, 0, 8, 8),
                child: Row(children: [
                  IconButton(
                    tooltip: fav ? 'Retirer des favoris' : 'Ajouter aux favoris',
                    icon: Icon(fav ? Icons.favorite : Icons.favorite_border, color: fav ? Colors.redAccent : null),
                    onPressed: () => s.toggleFavorite(a.id),
                  ),
                  IconButton(
                    tooltip: got ? 'Dans ta collection' : 'Ajouter à ma collection',
                    icon: Icon(got ? Icons.check_circle : Icons.add_circle_outline,
                        color: got ? theme.colorScheme.primary : null),
                    onPressed: got ? null : () => showAdded(context, s, a, s.add(a, via: 'wildcard')),
                  ),
                  const Spacer(),
                  TextButton.icon(
                    onPressed: () => openArtwork(context, s, a, list: _items),
                    icon: const Icon(Icons.menu_book_outlined),
                    label: const Text('Voir la fiche'),
                  ),
                ]),
              ),
            ]),
          ),
        ),
      ),
    );
  }
}
