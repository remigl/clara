import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'artwork_screen.dart';
import 'core.dart';
import 'wildcards.dart';

// ---------------------------------------------------------------------------- coquille

class HomeShell extends StatefulWidget {
  const HomeShell({super.key, required this.state});
  final AppState state;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    // La caméra n'existe que sur l'onglet Scanner (libérée ailleurs, pour la batterie).
    final body = switch (_tab) {
      0 => ScanScreen(state: s, onOpenMuseums: () => setState(() => _tab = 3)),
      1 => CollectionScreen(state: s),
      2 => WildcardsScreen(state: s, onOpenMuseums: () => setState(() => _tab = 3)),
      _ => MuseumsScreen(state: s),
    };
    return Scaffold(
      body: body,
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.center_focus_strong_outlined), label: 'Scanner'),
          NavigationDestination(icon: Icon(Icons.collections_outlined), label: 'Ma collection'),
          NavigationDestination(icon: Icon(Icons.style_outlined), label: 'Wildcards'),
          NavigationDestination(icon: Icon(Icons.museum_outlined), label: 'Musées'),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------- éléments communs

/// Vignette d'une œuvre ; [cover] = photo personnelle à montrer à la place de l'image du musée.
Widget artThumb(Artwork a, {double? width, double? height, BoxFit fit = BoxFit.cover, File? cover}) {
  Widget placeholder() => Container(
        width: width,
        height: height,
        color: const Color(0xFFEDE6DD),
        alignment: Alignment.center,
        child: const Icon(Icons.image_not_supported_outlined, color: Color(0xFF9C8B78)),
      );
  final f = cover ?? a.thumbFile;
  if (f == null) return placeholder();
  return Image.file(f, width: width, height: height, fit: fit, errorBuilder: (_, __, ___) => placeholder());
}

class ProgressBlock extends StatelessWidget {
  const ProgressBlock({super.key, required this.state, required this.pack});
  final AppState state;
  final MuseumPack pack;

  @override
  Widget build(BuildContext context) {
    final n = state.collectedCount(pack);
    final p = state.progress(pack);
    final pct = (p * 100).toStringAsFixed(p < 0.1 && n > 0 ? 1 : 0);
    final what = pack.knowsDisplay ? 'œuvres exposées' : 'œuvres';
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      ClipRRect(
        borderRadius: BorderRadius.circular(6),
        child: LinearProgressIndicator(value: p, minHeight: 8),
      ),
      const SizedBox(height: 6),
      Text('$n / ${pack.numberOfWorks} $what découvertes · $pct %', style: Theme.of(context).textTheme.bodyMedium),
    ]);
  }
}

void showAdded(BuildContext context, AppState s, Artwork a, bool isNew) {
  final p = a.pack;
  final pct = (s.progress(p) * 100).toStringAsFixed(1);
  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
    content: Text(isNew
        ? 'Nouvelle œuvre : ${a.title} — ${s.collectedCount(p)}/${p.numberOfWorks} à ${p.name} ($pct %)'
        : 'Déjà dans ta collection : ${a.title}'),
  ));
}

/// Regroupe des éléments par pays puis par ville, dans l'ordre alphabétique français.
List<(String, String, List<List<T>>)> groupByCountryCity<T>(
    List<T> items, String Function(T) country, String Function(T) city, String Function(T) name) {
  final byCountry = <String, Map<String, List<T>>>{};
  for (final it in items) {
    byCountry.putIfAbsent(country(it), () => {}).putIfAbsent(city(it), () => []).add(it);
  }
  final countries = byCountry.keys.toList()..sort((a, b) => sortKey(a).compareTo(sortKey(b)));
  return [
    for (final c in countries)
      () {
        final cities = byCountry[c]!.keys.toList()..sort((a, b) => sortKey(a).compareTo(sortKey(b)));
        return (
          c,
          cities.join(', '),
          [
            for (final ci in cities)
              byCountry[c]![ci]!..sort((a, b) => sortKey(name(a)).compareTo(sortKey(name(b))))
          ]
        );
      }()
  ];
}

// ---------------------------------------------------------------------------- scanner

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key, required this.state, required this.onOpenMuseums});
  final AppState state;
  final VoidCallback onOpenMuseums;

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  String? _error;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        setState(() => _error = 'Aucune caméra trouvée.');
        return;
      }
      final back = cams.firstWhere((c) => c.lensDirection == CameraLensDirection.back, orElse: () => cams.first);
      final c = CameraController(back, ResolutionPreset.high, enableAudio: false);
      await c.initialize();
      if (!mounted) {
        await c.dispose();
        return;
      }
      setState(() {
        _controller = c;
        _error = null;
      });
    } on CameraException catch (e) {
      setState(() => _error = e.code == 'CameraAccessDenied'
          ? 'Clara a besoin de la caméra pour reconnaître les œuvres. Autorise-la dans les réglages du téléphone.'
          : 'Caméra indisponible : ${e.description ?? e.code}');
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final c = _controller;
    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      _controller = null;
      c?.dispose();
    } else if (state == AppLifecycleState.resumed && c == null) {
      _initCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _scan() async {
    final c = _controller;
    if (c == null || !c.value.isInitialized || _busy) return;
    if (widget.state.packs.isEmpty) {
      widget.onOpenMuseums();
      return;
    }
    setState(() => _busy = true);
    try {
      HapticFeedback.lightImpact();
      final shot = await c.takePicture();
      final Uint8List bytes = await shot.readAsBytes();
      final cands = await widget.state.recognize(bytes);
      if (!mounted) return;
      await _showResult(cands, bytes);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Échec de la reconnaissance : $e')));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _showResult(List<Candidate> cands, Uint8List photo) async {
    final s = widget.state;
    final museumId = s.current?.id;
    if (cands.isEmpty) {
      s.store.logScan(museumId, cands, 'no_match');
      return;
    }
    final chosen = await showModalBottomSheet<Artwork>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => ResultSheet(state: s, candidates: cands, showMuseum: s.current == null),
    );
    if (!mounted) return;
    if (chosen == null) {
      s.store.logScan(museumId, cands, 'rejected');
      return;
    }
    final isNew = s.add(chosen);
    await s.savePhoto(chosen, photo); // la photo du scan est liée à l'œuvre (affichée sur sa fiche)
    if (!mounted) return;
    s.store.logScan(museumId, cands, chosen.id == cands.first.artwork.id ? 'user_confirmed' : 'user_picked_alt',
        chosen: chosen.id);
    HapticFeedback.mediumImpact();
    showAdded(context, s, chosen, isNew);
  }

  Future<void> _pickMuseum() async {
    await showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      isScrollControlled: true,
      builder: (_) => MuseumPicker(state: widget.state, onMore: widget.onOpenMuseums),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = _controller;
    final s = widget.state;
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) {
        final label = s.current?.name ?? 'Reconnaître partout';
        return Stack(fit: StackFit.expand, children: [
          Container(color: Colors.black),
          if (c != null && c.value.isInitialized)
            Center(child: AspectRatio(aspectRatio: 1 / c.value.aspectRatio, child: CameraPreview(c))),
          if (_error != null)
            Center(
              child: Padding(
                padding: const EdgeInsets.all(32),
                child: Text(_error!, style: const TextStyle(color: Colors.white), textAlign: TextAlign.center),
              ),
            ),
          IgnorePointer(
            child: Center(
              child: FractionallySizedBox(
                widthFactor: 0.8,
                heightFactor: 0.55,
                child: Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.white70, width: 2),
                    borderRadius: BorderRadius.circular(16),
                  ),
                ),
              ),
            ),
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.topCenter,
              child: Padding(
                padding: const EdgeInsets.only(top: 12),
                child: ActionChip(
                  avatar: Icon(s.current == null ? Icons.public : Icons.museum_outlined, size: 18),
                  label: Row(mainAxisSize: MainAxisSize.min, children: [
                    ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 240),
                      child: Text(label, overflow: TextOverflow.ellipsis),
                    ),
                    const Icon(Icons.arrow_drop_down, size: 20),
                  ]),
                  onPressed: _pickMuseum,
                ),
              ),
            ),
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 24),
                child: Column(mainAxisSize: MainAxisSize.min, children: [
                  const Text('Cadre le tableau, puis appuie',
                      style: TextStyle(color: Colors.white, shadows: [Shadow(blurRadius: 4)])),
                  const SizedBox(height: 12),
                  GestureDetector(
                    onTap: _scan,
                    child: Container(
                      width: 78,
                      height: 78,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white,
                        border: Border.all(color: Colors.white54, width: 6),
                      ),
                      alignment: Alignment.center,
                      child: _busy
                          ? const SizedBox(width: 32, height: 32, child: CircularProgressIndicator(strokeWidth: 3))
                          : const Icon(Icons.search, size: 34, color: Color(0xFF8C5A2B)),
                    ),
                  ),
                ]),
              ),
            ),
          ),
        ]);
      },
    );
  }
}

/// Choix du musée où l'on scanne : musées installés, par pays puis ville, ou « partout ».
class MuseumPicker extends StatelessWidget {
  const MuseumPicker({super.key, required this.state, required this.onMore});
  final AppState state;
  final VoidCallback onMore;

  @override
  Widget build(BuildContext context) {
    final s = state;
    final groups = groupByCountryCity<MuseumPack>(s.sortedPacks(), (p) => p.countryName, (p) => p.city, (p) => p.name);
    final theme = Theme.of(context);
    return SafeArea(
      child: ConstrainedBox(
        constraints: BoxConstraints(maxHeight: MediaQuery.of(context).size.height * 0.75),
        child: ListView(shrinkWrap: true, children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
            child: Text('Où es-tu ?', style: theme.textTheme.titleLarge),
          ),
          ListTile(
            leading: _radio(s.currentId == everywhere, theme),
            trailing: const Icon(Icons.public),
            title: const Text('Reconnaître partout'),
            subtitle: Text('Dans les ${s.packs.length} musées téléchargés'),
            onTap: () {
              s.setCurrent(everywhere);
              Navigator.pop(context);
            },
          ),
          for (final (country, _, cities) in groups) ...[
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 4),
              child: Text('${countryFlag(cities.first.first.country)}  $country', style: theme.textTheme.titleSmall),
            ),
            for (final museums in cities)
              for (final p in museums)
                ListTile(
                  leading: _radio(s.currentId == p.id, theme),
                  title: Text(p.name),
                  subtitle: Text(p.city),
                  onTap: () {
                    s.setCurrent(p.id);
                    Navigator.pop(context);
                  },
                ),
          ],
          ListTile(
            leading: const Icon(Icons.download_outlined),
            title: const Text('Télécharger d\'autres musées'),
            onTap: () {
              Navigator.pop(context);
              onMore();
            },
          ),
        ]),
      ),
    );
  }
}

Widget _radio(bool on, ThemeData theme) =>
    Icon(on ? Icons.radio_button_checked : Icons.radio_button_unchecked, color: theme.colorScheme.primary);

class ResultSheet extends StatelessWidget {
  const ResultSheet({super.key, required this.state, required this.candidates, required this.showMuseum});
  final AppState state;
  final List<Candidate> candidates;
  final bool showMuseum;

  @override
  Widget build(BuildContext context) {
    final best = candidates.first.artwork;
    final theme = Theme.of(context);
    final already = state.isCollected(best.id);
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('C\'est bien celle-ci ?', style: theme.textTheme.titleMedium),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: SizedBox(height: 220, child: artThumb(best, fit: BoxFit.contain)),
          ),
          const SizedBox(height: 12),
          Text(best.title, style: theme.textTheme.titleLarge),
          Text([best.artist, best.date].whereType<String>().join(' · '), style: theme.textTheme.bodyMedium),
          if (showMuseum || best.place.isNotEmpty)
            Text([if (showMuseum) best.pack.name, if (best.place.isNotEmpty) best.place].join(' · '),
                style: theme.textTheme.bodySmall),
          if (already) Text('Déjà dans ta collection', style: TextStyle(color: theme.colorScheme.primary)),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: () => Navigator.pop(context, best),
            icon: const Icon(Icons.check),
            label: Text(already ? 'Oui' : 'Oui, l\'ajouter à ma collection'),
          ),
          if (candidates.length > 1) ...[
            const SizedBox(height: 8),
            Text('Ou peut-être :', style: theme.textTheme.labelLarge),
            for (final c in candidates.skip(1))
              ListTile(
                contentPadding: EdgeInsets.zero,
                leading: ClipRRect(borderRadius: BorderRadius.circular(6), child: artThumb(c.artwork, width: 48, height: 48)),
                title: Text(c.artwork.title, maxLines: 1, overflow: TextOverflow.ellipsis),
                subtitle: Text(
                    [c.artwork.artist ?? '', if (showMuseum) c.artwork.pack.name].where((x) => x.isNotEmpty).join(' · '),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis),
                onTap: () => Navigator.pop(context, c.artwork),
              ),
          ],
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Aucune de celles-ci')),
        ]),
      ),
    );
  }
}

// ---------------------------------------------------------------------------- collection

class CollectionScreen extends StatelessWidget {
  const CollectionScreen({super.key, required this.state});
  final AppState state;

  Widget _tile(BuildContext context, Artwork a, List<Artwork> list, {bool showFav = false}) {
    final theme = Theme.of(context);
    return InkWell(
      onTap: () => openArtwork(context, state, a, list: list),
      borderRadius: BorderRadius.circular(8),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Expanded(
          child: Stack(fit: StackFit.expand, children: [
            ClipRRect(borderRadius: BorderRadius.circular(8), child: artThumb(a, cover: state.coverOf(a.id))),
            if (showFav && state.isFavorite(a.id))
              const Positioned(
                  right: 4, top: 4, child: Icon(Icons.favorite, color: Colors.white, size: 16, shadows: [Shadow(blurRadius: 3)])),
          ]),
        ),
        const SizedBox(height: 4),
        Text(a.title, maxLines: 1, overflow: TextOverflow.ellipsis, style: theme.textTheme.bodySmall),
      ]),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: state,
      builder: (context, _) {
        final counts = state.collectedByMuseum();
        final museums = state.sortedPacks().where((p) => (counts[p.id] ?? 0) > 0).toList();
        final favorites = state.favorites();
        final theme = Theme.of(context);
        final total = counts.values.fold<int>(0, (a, b) => a + b);
        return SafeArea(
          child: CustomScrollView(slivers: [
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
                child: Text('Ma collection', style: theme.textTheme.headlineSmall),
              ),
            ),
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Text(total == 0
                    ? ''
                    : '$total œuvre${total > 1 ? 's' : ''} dans ${counts.length} musée${counts.length > 1 ? 's' : ''}'),
              ),
            ),
            // --- favoris : apparaît dès le premier favori
            if (favorites.isNotEmpty) ...[
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 20, 16, 8),
                  child: Row(children: [
                    const Icon(Icons.favorite, color: Colors.redAccent, size: 20),
                    const SizedBox(width: 8),
                    Text('Favoris', style: theme.textTheme.titleMedium),
                    const SizedBox(width: 6),
                    Text('${favorites.length}', style: theme.textTheme.bodySmall),
                  ]),
                ),
              ),
              SliverToBoxAdapter(
                child: SizedBox(
                  height: 170,
                  child: ListView.separated(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    scrollDirection: Axis.horizontal,
                    itemCount: favorites.length,
                    separatorBuilder: (_, __) => const SizedBox(width: 10),
                    itemBuilder: (context, i) => SizedBox(width: 120, child: _tile(context, favorites[i], favorites)),
                  ),
                ),
              ),
            ],
            if (museums.isEmpty && favorites.isEmpty)
              const SliverFillRemaining(
                hasScrollBody: false,
                child: Center(
                  child: Padding(
                    padding: EdgeInsets.all(32),
                    child: Text('Aucune œuvre pour l\'instant.\nScanne un tableau pour commencer ta collection.',
                        textAlign: TextAlign.center),
                  ),
                ),
              ),
            for (final p in museums) ...[
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 24, 16, 8),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(p.name, style: theme.textTheme.titleMedium),
                    Text('${p.city}, ${p.countryName}', style: theme.textTheme.bodySmall),
                    const SizedBox(height: 8),
                    ProgressBlock(state: state, pack: p),
                  ]),
                ),
              ),
              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                sliver: Builder(builder: (context) {
                  final items = state.collectedIn(p);
                  return SliverGrid.builder(
                    gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 3, mainAxisSpacing: 8, crossAxisSpacing: 8, childAspectRatio: 0.75),
                    itemCount: items.length,
                    itemBuilder: (context, i) => _tile(context, items[i], items, showFav: true),
                  );
                }),
              ),
            ],
            const SliverToBoxAdapter(child: SizedBox(height: 24)),
          ]),
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------- musées

/// Une ligne de la liste des musées : installé (pack) et/ou proposé au téléchargement (catalogue).
class _MuseumEntry {
  _MuseumEntry(this.id, this.name, this.city, this.countryName, this.country, this.pack, this.remote);
  final String id, name, city, countryName, country;
  final MuseumPack? pack;
  final RemotePack? remote;
}

class MuseumsScreen extends StatefulWidget {
  const MuseumsScreen({super.key, required this.state});
  final AppState state;

  @override
  State<MuseumsScreen> createState() => _MuseumsScreenState();
}

class _MuseumsScreenState extends State<MuseumsScreen> {
  String _query = '';

  List<_MuseumEntry> _entries() {
    final s = widget.state;
    final m = <String, _MuseumEntry>{};
    for (final r in s.catalog ?? const <RemotePack>[]) {
      m[r.id] = _MuseumEntry(r.id, r.name, r.city, r.countryName, r.country, s.packs[r.id], r);
    }
    for (final p in s.packs.values) {
      m[p.id] = _MuseumEntry(p.id, p.name, p.city, p.countryName, p.country, p, m[p.id]?.remote);
    }
    final q = sortKey(_query.trim());
    return [
      for (final e in m.values)
        if (q.isEmpty || sortKey('${e.name} ${e.city} ${e.countryName}').contains(q)) e
    ];
  }

  Future<void> _download(RemotePack r) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      await widget.state.download(r);
      messenger.showSnackBar(SnackBar(content: Text('${r.name} est prêt : tu peux y scanner, même hors ligne.')));
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Téléchargement de ${r.name} impossible : $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    final theme = Theme.of(context);
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) {
        final entries = _entries();
        final byContinent = <String, List<_MuseumEntry>>{};
        for (final e in entries) {
          byContinent.putIfAbsent(continentOf(e.country), () => []).add(e);
        }
        final installed = entries.where((e) => e.pack != null).length;
        return SafeArea(
          child: RefreshIndicator(
            onRefresh: s.refreshCatalog,
            child: ListView(children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                child: Text('Musées', style: theme.textTheme.headlineSmall),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                child: TextField(
                  decoration: const InputDecoration(
                      prefixIcon: Icon(Icons.search), hintText: 'Musée, ville, pays…', border: OutlineInputBorder(),
                      isDense: true),
                  onChanged: (v) => setState(() => _query = v),
                ),
              ),
              if (s.catalogLoading) const LinearProgressIndicator(),
              if (s.catalogError != null)
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(
                      s.catalog == null
                          ? 'Liste des musées indisponible (${s.catalogError}) : seuls les musées déjà téléchargés '
                              'sont affichés. Tire vers le bas pour réessayer.'
                          : 'Liste des musées non actualisée (${s.catalogError}) : dernière liste connue affichée.',
                      style: theme.textTheme.bodySmall),
                ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                child: Text(
                    '${entries.length} musée${entries.length > 1 ? 's' : ''} · $installed sur ce téléphone',
                    style: theme.textTheme.bodySmall),
              ),
              for (final continent in continents)
                if (byContinent[continent] case final list?)
                  for (final (i, (country, _, cities)) in groupByCountryCity<_MuseumEntry>(
                          list, (e) => e.countryName, (e) => e.city, (e) => e.name)
                      .indexed) ...[
                if (i == 0)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 20, 16, 4),
                    child: Text(continent.toUpperCase(),
                        style: theme.textTheme.labelLarge
                            ?.copyWith(letterSpacing: 1.5, color: theme.colorScheme.primary)),
                  ),
                ExpansionTile(
                  initiallyExpanded: _query.isNotEmpty || cities.any((c) => c.any((e) => e.pack != null)),
                  leading: Text(countryFlag(cities.first.first.country), style: const TextStyle(fontSize: 24)),
                  title: Text(country, style: theme.textTheme.titleMedium),
                  subtitle: Text(() {
                    final n = cities.fold<int>(0, (a, c) => a + c.length);
                    return '$n musée${n > 1 ? 's' : ''}';
                  }()),
                  children: [
                    for (final museums in cities) ...[
                      Padding(
                        padding: const EdgeInsets.fromLTRB(20, 8, 16, 0),
                        child: Align(
                          alignment: Alignment.centerLeft,
                          child: Text(museums.first.city, style: theme.textTheme.labelLarge),
                        ),
                      ),
                      for (final e in museums) _museumTile(context, e),
                    ],
                    _downloadAll(context, [for (final c in cities) ...c]),
                  ],
                ),
              ],
              const SizedBox(height: 24),
            ]),
          ),
        );
      },
    );
  }

  /// « Tout télécharger » pour un pays, quand plusieurs de ses musées manquent.
  Widget _downloadAll(BuildContext context, List<_MuseumEntry> list) {
    final s = widget.state;
    final todo = [
      for (final e in list)
        if (e.pack == null && e.remote != null && s.isCompatible(e.remote!) && !s.downloads.containsKey(e.id)) e.remote!
    ];
    if (todo.length < 2) return const SizedBox.shrink();
    final bytes = todo.fold<int>(0, (a, r) => a + r.bytes);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 12),
      child: Align(
        alignment: Alignment.centerLeft,
        child: TextButton.icon(
          icon: const Icon(Icons.download_for_offline_outlined),
          label: Text('Télécharger les ${todo.length} musées (${formatBytes(bytes)})'),
          onPressed: () async {
            for (final r in todo) {
              await _download(r);
            }
          },
        ),
      ),
    );
  }

  Widget _museumTile(BuildContext context, _MuseumEntry e) {
    final s = widget.state;
    final r = e.remote;
    final p = e.pack;
    final progress = s.downloads[e.id];
    final theme = Theme.of(context);
    if (progress != null) {
      return ListTile(
        title: Text(e.name),
        subtitle: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const SizedBox(height: 6),
          LinearProgressIndicator(value: progress > 0 ? progress : null),
          const SizedBox(height: 4),
          Text('Téléchargement… ${(progress * 100).toStringAsFixed(0)} %'),
        ]),
      );
    }
    if (p != null) {
      final isCurrent = s.currentId == p.id;
      final update = r != null && s.hasUpdate(r);
      return ListTile(
        title: Text(p.name),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 6),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            ProgressBlock(state: s, pack: p),
            if (update)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text('Mise à jour disponible (${formatBytes(r.bytes)})',
                    style: TextStyle(color: theme.colorScheme.primary)),
              ),
          ]),
        ),
        trailing: PopupMenuButton<String>(
          onSelected: (v) async {
            if (v == 'scan') s.setCurrent(p.id);
            if (v == 'update' && r != null) await _download(r);
            if (v == 'delete') {
              final ok = await showDialog<bool>(
                context: context,
                builder: (_) => AlertDialog(
                  title: Text('Supprimer ${p.name} ?'),
                  content: const Text('Ta collection est conservée ; tu pourras retélécharger le musée.'),
                  actions: [
                    TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Annuler')),
                    FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Supprimer')),
                  ],
                ),
              );
              if (ok == true) s.remove(p.id);
            }
          },
          itemBuilder: (_) => [
            if (!isCurrent) const PopupMenuItem(value: 'scan', child: Text('Scanner dans ce musée')),
            if (r != null && s.hasUpdate(r)) const PopupMenuItem(value: 'update', child: Text('Mettre à jour')),
            const PopupMenuItem(value: 'delete', child: Text('Supprimer du téléphone')),
          ],
        ),
        leading: Icon(isCurrent ? Icons.radio_button_checked : Icons.check_circle_outline,
            color: theme.colorScheme.primary),
        onTap: () => Navigator.of(context)
            .push(MaterialPageRoute(builder: (_) => MuseumDetailScreen(state: s, pack: p))),
      );
    }
    final compatible = r != null && s.isCompatible(r);
    return ListTile(
      title: Text(e.name),
      subtitle: Text(r == null
          ? ''
          : compatible
              ? '${r.works} œuvres · ${formatBytes(r.bytes)}'
              : 'Nécessite une version plus récente de Clara'),
      trailing: IconButton.filledTonal(
        icon: const Icon(Icons.download),
        tooltip: 'Télécharger',
        onPressed: compatible ? () => _download(r) : null,
      ),
    );
  }
}

class MuseumDetailScreen extends StatefulWidget {
  const MuseumDetailScreen({super.key, required this.state, required this.pack});
  final AppState state;
  final MuseumPack pack;

  @override
  State<MuseumDetailScreen> createState() => _MuseumDetailScreenState();
}

class _MuseumDetailScreenState extends State<MuseumDetailScreen> {
  String _query = '';

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    final p = widget.pack;
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) {
        final items = _query.trim().isEmpty ? p.all() : p.searchText(_query);
        final isCurrent = s.currentId == p.id;
        return Scaffold(
          appBar: AppBar(title: Text(p.name, maxLines: 1, overflow: TextOverflow.ellipsis)),
          body: Column(children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('${countryFlag(p.country)}  ${p.city}, ${p.countryName}'),
                const SizedBox(height: 8),
                ProgressBlock(state: s, pack: p),
                if (!p.knowsDisplay) ...[
                  const SizedBox(height: 6),
                  Text('Ce musée ne publie pas la liste des œuvres exposées : toutes ses peintures connues sont '
                      'listées, certaines peuvent être en réserve.',
                      style: Theme.of(context).textTheme.bodySmall),
                ],
                const SizedBox(height: 8),
                if (!isCurrent)
                  FilledButton.tonalIcon(
                    icon: const Icon(Icons.center_focus_strong_outlined),
                    label: const Text('Scanner dans ce musée'),
                    onPressed: () => s.setCurrent(p.id),
                  ),
              ]),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: TextField(
                decoration: const InputDecoration(
                    prefixIcon: Icon(Icons.search), hintText: 'Titre, artiste…', border: OutlineInputBorder(),
                    isDense: true),
                onChanged: (v) => setState(() => _query = v),
              ),
            ),
            Expanded(
              child: ListView.builder(
                itemCount: items.length,
                itemBuilder: (context, i) {
                  final a = items[i];
                  final got = s.isCollected(a.id);
                  return ListTile(
                    leading: ClipRRect(
                        borderRadius: BorderRadius.circular(6),
                        child: artThumb(a, width: 52, height: 52, cover: s.coverOf(a.id))),
                    title: Text(a.title, maxLines: 1, overflow: TextOverflow.ellipsis),
                    subtitle: Text([a.artist ?? 'Anonyme', if (a.place.isNotEmpty) a.place].join(' · '),
                        maxLines: 1, overflow: TextOverflow.ellipsis),
                    trailing: got
                        ? Icon(Icons.check_circle, color: Theme.of(context).colorScheme.primary)
                        : const Icon(Icons.radio_button_unchecked, color: Colors.black26),
                    onTap: () => openArtwork(context, s, a, list: items),
                  );
                },
              ),
            ),
          ]),
        );
      },
    );
  }
}

