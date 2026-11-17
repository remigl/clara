import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'core.dart';

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
      0 => ScanScreen(state: s),
      1 => CollectionScreen(state: s),
      _ => MuseumScreen(state: s),
    };
    return Scaffold(
      body: body,
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.center_focus_strong_outlined), label: 'Scanner'),
          NavigationDestination(icon: Icon(Icons.collections_outlined), label: 'Ma collection'),
          NavigationDestination(icon: Icon(Icons.museum_outlined), label: 'Musée'),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------- éléments communs

Widget artThumb(Artwork a, {double? width, double? height, BoxFit fit = BoxFit.cover}) {
  Widget placeholder() => Container(
        width: width,
        height: height,
        color: const Color(0xFFEDE6DD),
        alignment: Alignment.center,
        child: const Icon(Icons.image_not_supported_outlined, color: Color(0xFF9C8B78)),
      );
  final asset = a.thumbAsset;
  if (asset == null) return placeholder();
  return Image.asset(asset, width: width, height: height, fit: fit, errorBuilder: (_, __, ___) => placeholder());
}

class ProgressCard extends StatelessWidget {
  const ProgressCard({super.key, required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) {
    final p = state.pack;
    final n = state.collectedCount;
    final pct = (state.progress * 100).toStringAsFixed(state.progress < 0.1 && n > 0 ? 1 : 0);
    final theme = Theme.of(context);
    return Card(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(p.name, style: theme.textTheme.titleLarge),
          if (p.city.isNotEmpty) Text(p.city, style: theme.textTheme.bodyMedium),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(value: state.progress, minHeight: 10),
          ),
          const SizedBox(height: 8),
          Text('$n / ${p.numberOfWorks} œuvres découvertes · $pct %', style: theme.textTheme.bodyLarge),
        ]),
      ),
    );
  }
}

void showAdded(BuildContext context, AppState s, Artwork a, bool isNew) {
  final pct = (s.progress * 100).toStringAsFixed(1);
  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
    content: Text(isNew
        ? 'Nouvelle œuvre : ${a.title} — ${s.collectedCount}/${s.pack.numberOfWorks} ($pct %)'
        : 'Déjà dans ta collection : ${a.title}'),
  ));
}

// ---------------------------------------------------------------------------- scanner

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key, required this.state});
  final AppState state;

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
    setState(() => _busy = true);
    try {
      HapticFeedback.lightImpact();
      final shot = await c.takePicture();
      final Uint8List bytes = await shot.readAsBytes();
      final cands = await widget.state.recognize(bytes);
      if (!mounted) return;
      await _showResult(cands);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Échec de la reconnaissance : $e')));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _showResult(List<Candidate> cands) async {
    final s = widget.state;
    if (cands.isEmpty) {
      s.store.logScan(s.pack.museumId, cands, 'no_match');
      return;
    }
    final chosen = await showModalBottomSheet<Artwork>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => ResultSheet(state: s, candidates: cands),
    );
    if (!mounted) return;
    if (chosen == null) {
      s.store.logScan(s.pack.museumId, cands, 'rejected');
      return;
    }
    final isNew = s.add(chosen.id);
    s.store.logScan(s.pack.museumId, cands, chosen.id == cands.first.artwork.id ? 'user_confirmed' : 'user_picked_alt',
        chosen: chosen.id);
    HapticFeedback.mediumImpact();
    showAdded(context, s, chosen, isNew);
  }

  @override
  Widget build(BuildContext context) {
    final c = _controller;
    return Stack(fit: StackFit.expand, children: [
      Container(color: Colors.black),
      if (c != null && c.value.isInitialized)
        Center(
          child: AspectRatio(
            aspectRatio: 1 / c.value.aspectRatio,
            child: CameraPreview(c),
          ),
        ),
      if (_error != null)
        Center(
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Text(_error!, style: const TextStyle(color: Colors.white), textAlign: TextAlign.center),
          ),
        ),
      // cadre de visée
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
            child: Chip(
              avatar: const Icon(Icons.museum_outlined, size: 18),
              label: Text(widget.state.pack.name),
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
  }
}

class ResultSheet extends StatelessWidget {
  const ResultSheet({super.key, required this.state, required this.candidates});
  final AppState state;
  final List<Candidate> candidates;

  @override
  Widget build(BuildContext context) {
    final best = candidates.first.artwork;
    final theme = Theme.of(context);
    final already = state.isCollected(best.id);
    return SafeArea(
      child: Padding(
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
          Text(best.place, style: theme.textTheme.bodySmall),
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
                leading: ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: artThumb(c.artwork, width: 48, height: 48),
                ),
                title: Text(c.artwork.title, maxLines: 1, overflow: TextOverflow.ellipsis),
                subtitle: Text(c.artwork.artist ?? '', maxLines: 1, overflow: TextOverflow.ellipsis),
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

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: state,
      builder: (context, _) {
        final items = state.collectedArtworks();
        return SafeArea(
          child: CustomScrollView(slivers: [
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                child: Text('Ma collection', style: Theme.of(context).textTheme.headlineSmall),
              ),
            ),
            SliverToBoxAdapter(child: ProgressCard(state: state)),
            if (items.isEmpty)
              const SliverFillRemaining(
                hasScrollBody: false,
                child: Center(
                  child: Padding(
                    padding: EdgeInsets.all(32),
                    child: Text('Aucune œuvre pour l\'instant.\nScanne un tableau pour commencer ta collection.',
                        textAlign: TextAlign.center),
                  ),
                ),
              )
            else
              SliverPadding(
                padding: const EdgeInsets.all(12),
                sliver: SliverGrid.builder(
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 3, mainAxisSpacing: 8, crossAxisSpacing: 8, childAspectRatio: 0.75),
                  itemCount: items.length,
                  itemBuilder: (context, i) {
                    final a = items[i];
                    return InkWell(
                      onTap: () => openArtwork(context, state, a),
                      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                        Expanded(
                          child: ClipRRect(borderRadius: BorderRadius.circular(8), child: artThumb(a)),
                        ),
                        const SizedBox(height: 4),
                        Text(a.title, maxLines: 1, overflow: TextOverflow.ellipsis,
                            style: Theme.of(context).textTheme.bodySmall),
                      ]),
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

// ---------------------------------------------------------------------------- musée

class MuseumScreen extends StatefulWidget {
  const MuseumScreen({super.key, required this.state});
  final AppState state;

  @override
  State<MuseumScreen> createState() => _MuseumScreenState();
}

class _MuseumScreenState extends State<MuseumScreen> {
  String _query = '';

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) {
        final items = _query.trim().isEmpty ? s.pack.all() : s.pack.searchText(_query);
        return SafeArea(
          child: Column(children: [
            ProgressCard(state: s),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: TextField(
                decoration: const InputDecoration(
                  prefixIcon: Icon(Icons.search),
                  hintText: 'Titre, artiste…',
                  border: OutlineInputBorder(),
                  isDense: true,
                ),
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
                        borderRadius: BorderRadius.circular(6), child: artThumb(a, width: 52, height: 52)),
                    title: Text(a.title, maxLines: 1, overflow: TextOverflow.ellipsis),
                    subtitle: Text('${a.artist ?? 'Anonyme'} · ${a.place}', maxLines: 1, overflow: TextOverflow.ellipsis),
                    trailing: got
                        ? Icon(Icons.check_circle, color: Theme.of(context).colorScheme.primary)
                        : const Icon(Icons.radio_button_unchecked, color: Colors.black26),
                    onTap: () => openArtwork(context, s, a),
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

// ---------------------------------------------------------------------------- fiche

void openArtwork(BuildContext context, AppState s, Artwork a) {
  Navigator.of(context).push(MaterialPageRoute(builder: (_) => ArtworkScreen(state: s, artwork: a)));
}

class ArtworkScreen extends StatelessWidget {
  const ArtworkScreen({super.key, required this.state, required this.artwork});
  final AppState state;
  final Artwork artwork;

  @override
  Widget build(BuildContext context) {
    final a = artwork;
    final theme = Theme.of(context);
    return ListenableBuilder(
      listenable: state,
      builder: (context, _) {
        final got = state.isCollected(a.id);
        final fav = state.store.isFavorite(a.id);
        String? dims;
        if (a.heightCm != null && a.widthCm != null) {
          dims = '${a.heightCm!.toStringAsFixed(0)} × ${a.widthCm!.toStringAsFixed(0)} cm';
        }
        final facts = <(String, String?)>[
          ('Date', a.date),
          ('Technique', a.medium),
          ('Dimensions', dims),
          ('Emplacement', a.place),
          ('Inventaire', a.inventory),
        ];
        return Scaffold(
          appBar: AppBar(
            title: Text(a.artist ?? 'Œuvre', maxLines: 1, overflow: TextOverflow.ellipsis),
            actions: [
              IconButton(
                tooltip: 'Favori',
                icon: Icon(fav ? Icons.favorite : Icons.favorite_border),
                onPressed: () => state.toggleFavorite(a.id),
              ),
            ],
          ),
          body: ListView(padding: const EdgeInsets.all(16), children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: SizedBox(height: 300, child: artThumb(a, fit: BoxFit.contain)),
            ),
            const SizedBox(height: 16),
            Text(a.title, style: theme.textTheme.headlineSmall),
            if (a.artist != null) Text(a.artist!, style: theme.textTheme.titleMedium),
            const SizedBox(height: 12),
            for (final (label, value) in facts)
              if (value != null && value.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 3),
                  child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    SizedBox(width: 110, child: Text(label, style: theme.textTheme.labelLarge)),
                    Expanded(child: Text(value)),
                  ]),
                ),
            const SizedBox(height: 16),
            if (got)
              Row(children: [
                Icon(Icons.check_circle, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                const Text('Dans ta collection'),
              ])
            else
              OutlinedButton.icon(
                icon: const Icon(Icons.add),
                label: const Text('Je l\'ai vue : ajouter à ma collection'),
                onPressed: () {
                  final isNew = state.add(a.id, via: 'manual');
                  showAdded(context, state, a, isNew);
                },
              ),
            if (a.credit != null) ...[
              const SizedBox(height: 24),
              Text(a.credit!, style: theme.textTheme.bodySmall),
            ],
          ]),
        );
      },
    );
  }
}
