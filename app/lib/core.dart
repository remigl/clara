// Cœur hors ligne de Clara : pack musée (SQLite + vecteurs), modèle DINOv2 (ONNX Runtime),
// collection de l'utilisateur. Même logique que la référence Python (pipeline/clara_p0).
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqlite3/sqlite3.dart';

const String packId = 'dk-smk';
const int inputSize = 224;

// ---------------------------------------------------------------------------- œuvres

class Artwork {
  Artwork.fromRow(Row r)
      : id = r['id'] as String,
        museumId = r['museum_id'] as String,
        title = (r['title'] as String?) ?? 'Sans titre',
        artist = r['artist_display'] as String?,
        date = r['date_display'] as String?,
        medium = r['medium'] as String?,
        room = r['room'] as String?,
        status = (r['exhibition_status'] as String?) ?? 'unknown',
        inventory = r['inventory_number'] as String?,
        heightCm = (r['height_cm'] as num?)?.toDouble(),
        widthCm = (r['width_cm'] as num?)?.toDouble(),
        sourceUrl = r['source_url'] as String?,
        thumb = r['local_path'] as String?,
        credit = r['credit_line'] as String?;

  final String id;
  final String museumId;
  final String title;
  final String? artist;
  final String? date;
  final String? medium;
  final String? room;
  final String status;
  final String? inventory;
  final double? heightCm;
  final double? widthCm;
  final String? sourceUrl;
  final String? thumb;
  final String? credit;

  String? get thumbAsset => thumb == null ? null : 'assets/packs/$packId/$thumb';

  String get place {
    if (room != null && room!.isNotEmpty) return room!;
    switch (status) {
      case 'on_display':
        return 'Exposée';
      case 'not_on_display':
        return 'Non exposée actuellement';
      case 'temporarily_absent':
        return 'Temporairement absente';
      default:
        return 'Emplacement inconnu';
    }
  }
}

class Candidate {
  Candidate(this.artwork, this.score);
  final Artwork artwork;
  final double score;
}

const String _artworkSelect =
    'SELECT a.*, i.local_path, i.credit_line FROM artwork a LEFT JOIN artwork_image i ON i.artwork_id = a.id';

// ---------------------------------------------------------------------------- pack musée

class MuseumPack {
  MuseumPack._(this.manifest, this.db, this.dim, this.vectors, this.rowArtwork, this.museum);

  final Map<String, dynamic> manifest;
  final Database db;
  final int dim;
  final Float32List vectors; // lignes déquantifiées, normalisées L2
  final List<String> rowArtwork;
  final Map<String, Object?> museum;

  String get museumId => museum['id'] as String;
  String get name => (museum['name_local'] as String?) ?? (museum['name'] as String);
  String get city => (museum['city'] as String?) ?? '';
  int get numberOfWorks => (museum['number_of_works'] as int?) ?? 0;

  static Future<MuseumPack> load() async {
    const base = 'assets/packs/$packId';
    final manifest = jsonDecode(await rootBundle.loadString('$base/manifest.json')) as Map<String, dynamic>;
    final dir = await getApplicationSupportDirectory();
    final packDir = Directory('${dir.path}/packs/$packId');
    final dbFile = File('${packDir.path}/catalog.db');
    final stamp = File('${packDir.path}/.version-${manifest['version']}-${manifest['built_at']}');
    if (!stamp.existsSync() || !dbFile.existsSync()) {
      // SQLite a besoin d'un vrai fichier : copie unique depuis les assets (puis à chaque nouvelle version)
      packDir.createSync(recursive: true);
      final data = await rootBundle.load('$base/catalog.db');
      await dbFile.writeAsBytes(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes), flush: true);
      await stamp.writeAsString('ok');
    }
    final db = sqlite3.open(dbFile.path, mode: OpenMode.readOnly);
    final dim = manifest['embedding_dim'] as int;

    final q = await rootBundle.load('$base/vectors.bin');
    final qi = q.buffer.asInt8List(q.offsetInBytes, q.lengthInBytes);
    final s = await rootBundle.load('$base/scales.bin');
    final sd = ByteData.sublistView(s.buffer.asUint8List(s.offsetInBytes, s.lengthInBytes));
    final n = sd.lengthInBytes ~/ 4;
    final vectors = Float32List(n * dim);
    for (var i = 0; i < n; i++) {
      final scale = sd.getFloat32(i * 4, Endian.little);
      var norm = 0.0;
      for (var j = 0; j < dim; j++) {
        final v = qi[i * dim + j] * scale;
        vectors[i * dim + j] = v;
        norm += v * v;
      }
      norm = math.sqrt(norm);
      if (norm > 0) {
        for (var j = 0; j < dim; j++) {
          vectors[i * dim + j] /= norm;
        }
      }
    }
    final rows = db.select('SELECT row, artwork_id FROM artwork_embedding ORDER BY row');
    final rowArtwork = [for (final r in rows) r['artwork_id'] as String];
    if (rowArtwork.length != n) {
      throw StateError('Pack incohérent : ${rowArtwork.length} lignes, $n vecteurs');
    }
    final museum = Map<String, Object?>.from(db.select('SELECT * FROM museum LIMIT 1').first);
    return MuseumPack._(manifest, db, dim, vectors, rowArtwork, museum);
  }

  Artwork? artwork(String id) {
    final r = db.select('$_artworkSelect WHERE a.id = ?', [id]);
    return r.isEmpty ? null : Artwork.fromRow(r.first);
  }

  List<Artwork> all() =>
      [for (final r in db.select('$_artworkSelect ORDER BY a.room IS NULL, a.room, a.title')) Artwork.fromRow(r)];

  List<Artwork> searchText(String text) {
    final terms = text
        .toLowerCase()
        .split(RegExp(r'[^\p{L}\p{N}]+', unicode: true))
        .where((t) => t.isNotEmpty)
        .map((t) => '"$t"*')
        .join(' ');
    if (terms.isEmpty) return all();
    final r = db.select(
        'SELECT a.*, i.local_path, i.credit_line FROM artwork_fts f JOIN artwork a ON a.rowid = f.rowid '
        'LEFT JOIN artwork_image i ON i.artwork_id = a.id WHERE artwork_fts MATCH ? LIMIT 100',
        [terms]);
    return [for (final row in r) Artwork.fromRow(row)];
  }

  /// Plus proches œuvres : max sur les vues de la requête et les vues de référence d'une même œuvre.
  List<Candidate> nearest(List<Float32List> queries, {int k = 3}) {
    for (final q in queries) {
      if (q.length != dim) throw StateError('Dimension du modèle ${q.length} ≠ pack $dim');
    }
    final best = <String, double>{};
    for (var i = 0; i < rowArtwork.length; i++) {
      final off = i * dim;
      var m = -2.0;
      for (final q in queries) {
        var s = 0.0;
        for (var j = 0; j < dim; j++) {
          s += q[j] * vectors[off + j];
        }
        if (s > m) m = s;
      }
      final id = rowArtwork[i];
      final prev = best[id];
      if (prev == null || m > prev) best[id] = m;
    }
    final sorted = best.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    final out = <Candidate>[];
    for (final e in sorted.take(k)) {
      final a = artwork(e.key);
      if (a != null) out.add(Candidate(a, e.value));
    }
    return out;
  }
}

// ---------------------------------------------------------------------------- modèle

Float32List l2normalize(List<double> v) {
  var n = 0.0;
  for (final x in v) {
    n += x * x;
  }
  n = math.sqrt(n);
  final out = Float32List(v.length);
  for (var i = 0; i < v.length; i++) {
    out[i] = n > 0 ? v[i] / n : 0;
  }
  return out;
}

/// Prépare les vues d'une photo (complète + centre), au format DINOv2 : 224×224, RGB, normalisation ImageNet.
/// Fonction de premier niveau : exécutée dans un isolate (compute) pour ne pas bloquer l'interface.
List<Float32List> prepareViews(Uint8List jpeg) {
  var im = img.decodeImage(jpeg);
  if (im == null) return [];
  im = img.bakeOrientation(im);
  if (math.max(im.width, im.height) > 1024) {
    im = im.width >= im.height
        ? img.copyResize(im, width: 1024, interpolation: img.Interpolation.average)
        : img.copyResize(im, height: 1024, interpolation: img.Interpolation.average);
  }
  final w = im.width, h = im.height;
  final center = img.copyCrop(im,
      x: (w * 0.15).round(), y: (h * 0.15).round(), width: (w * 0.7).round(), height: (h * 0.7).round());
  return [_toTensor(im), _toTensor(center)];
}

Float32List _toTensor(img.Image src) {
  final scale = inputSize / math.min(src.width, src.height);
  final nw = math.max(inputSize, (src.width * scale).round());
  final nh = math.max(inputSize, (src.height * scale).round());
  final r = img.copyResize(src, width: nw, height: nh, interpolation: img.Interpolation.average);
  final x0 = (nw - inputSize) ~/ 2, y0 = (nh - inputSize) ~/ 2;
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const plane = inputSize * inputSize;
  final out = Float32List(3 * plane);
  for (var y = 0; y < inputSize; y++) {
    for (var x = 0; x < inputSize; x++) {
      final p = r.getPixel(x0 + x, y0 + y);
      final i = y * inputSize + x;
      out[i] = ((p.r / 255.0) - mean[0]) / std[0];
      out[plane + i] = ((p.g / 255.0) - mean[1]) / std[1];
      out[2 * plane + i] = ((p.b / 255.0) - mean[2]) / std[2];
    }
  }
  return out;
}

class Embedder {
  Embedder._(this._session);
  final OrtSession _session;

  static Future<Embedder> load() async {
    OrtEnv.instance.init();
    final raw = await rootBundle.load('assets/model/model.onnx');
    final bytes = raw.buffer.asUint8List(raw.offsetInBytes, raw.lengthInBytes);
    return Embedder._(OrtSession.fromBuffer(bytes, OrtSessionOptions()));
  }

  /// Empreinte « CLS » de DINOv2, normalisée L2 (réglage par défaut de la référence Python).
  Future<Float32List> embed(Float32List chw) async {
    final input = OrtValueTensor.createTensorWithDataList(chw, [1, 3, inputSize, inputSize]);
    final runOptions = OrtRunOptions();
    final outputs = await _session.runAsync(runOptions, {'pixel_values': input}) ?? <OrtValue?>[];
    input.release();
    runOptions.release();
    try {
      for (final o in outputs) {
        final v = o?.value;
        // last_hidden_state : [1, 1 + patchs, dim] -> jeton CLS = [0][0]
        if (v is List && v.isNotEmpty && v[0] is List && (v[0] as List).isNotEmpty && (v[0] as List)[0] is List) {
          final cls = ((v[0] as List)[0] as List).map((e) => (e as num).toDouble()).toList();
          return l2normalize(cls);
        }
      }
      final v = outputs.first?.value as List; // repli : sortie 2D [1, dim]
      return l2normalize((v[0] as List).map((e) => (e as num).toDouble()).toList());
    } finally {
      for (final o in outputs) {
        o?.release();
      }
    }
  }
}

// ---------------------------------------------------------------------------- collection

class UserStore {
  UserStore._(this.db);
  final Database db;

  static Future<UserStore> open() async {
    final dir = await getApplicationSupportDirectory();
    final db = sqlite3.open('${dir.path}/user.db');
    db.execute('''CREATE TABLE IF NOT EXISTS collected (
      artwork_id TEXT PRIMARY KEY, museum_id TEXT NOT NULL, first_seen_at TEXT NOT NULL,
      last_seen_at TEXT NOT NULL, times_seen INTEGER NOT NULL DEFAULT 1, via TEXT NOT NULL)''');
    db.execute('''CREATE TABLE IF NOT EXISTS favorite (artwork_id TEXT PRIMARY KEY, created_at TEXT NOT NULL)''');
    db.execute('''CREATE TABLE IF NOT EXISTS scan (
      id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, museum_id TEXT, artwork_id TEXT,
      score REAL, decision TEXT NOT NULL, candidates_json TEXT)''');
    return UserStore._(db);
  }

  static String _now() => DateTime.now().toUtc().toIso8601String();

  /// Retourne true si l'œuvre est nouvelle dans la collection.
  bool collect(String artworkId, String museumId, String via) {
    final exists = db.select('SELECT 1 FROM collected WHERE artwork_id = ?', [artworkId]).isNotEmpty;
    if (exists) {
      db.execute('UPDATE collected SET last_seen_at = ?, times_seen = times_seen + 1 WHERE artwork_id = ?',
          [_now(), artworkId]);
      return false;
    }
    db.execute('INSERT INTO collected VALUES (?, ?, ?, ?, 1, ?)', [artworkId, museumId, _now(), _now(), via]);
    return true;
  }

  List<String> collectedIds() =>
      [for (final r in db.select('SELECT artwork_id FROM collected ORDER BY last_seen_at DESC')) r['artwork_id'] as String];

  bool isFavorite(String id) => db.select('SELECT 1 FROM favorite WHERE artwork_id = ?', [id]).isNotEmpty;

  void toggleFavorite(String id) {
    if (isFavorite(id)) {
      db.execute('DELETE FROM favorite WHERE artwork_id = ?', [id]);
    } else {
      db.execute('INSERT INTO favorite VALUES (?, ?)', [id, _now()]);
    }
  }

  void logScan(String museumId, List<Candidate> c, String decision, {String? chosen}) {
    db.execute('INSERT INTO scan (created_at, museum_id, artwork_id, score, decision, candidates_json) VALUES (?,?,?,?,?,?)', [
      _now(),
      museumId,
      chosen,
      c.isEmpty ? null : c.first.score,
      decision,
      jsonEncode([for (final x in c) {'id': x.artwork.id, 'score': x.score}]),
    ]);
  }
}

// ---------------------------------------------------------------------------- état de l'app

class AppState extends ChangeNotifier {
  AppState._(this.pack, this.embedder, this.store) {
    _collected = store.collectedIds();
  }

  final MuseumPack pack;
  final Embedder embedder;
  final UserStore store;
  late List<String> _collected;

  static Future<AppState> load() async {
    final pack = await MuseumPack.load();
    final embedder = await Embedder.load();
    final store = await UserStore.open();
    return AppState._(pack, embedder, store);
  }

  List<String> get collectedIds => _collected;
  bool isCollected(String id) => _collected.contains(id);

  int get collectedCount {
    final ids = _collected.toSet();
    final r = pack.db.select('SELECT id FROM artwork WHERE counts_for_progress = 1');
    return r.where((row) => ids.contains(row['id'])).length;
  }

  double get progress => pack.numberOfWorks == 0 ? 0 : collectedCount / pack.numberOfWorks;

  List<Artwork> collectedArtworks() => [
        for (final id in _collected)
          if (pack.artwork(id) != null) pack.artwork(id)!
      ];

  Future<List<Candidate>> recognize(Uint8List jpeg) async {
    final views = await compute(prepareViews, jpeg);
    final q = <Float32List>[];
    for (final v in views) {
      q.add(await embedder.embed(v));
    }
    return pack.nearest(q, k: 3);
  }

  bool add(String artworkId, {String via = 'scan'}) {
    final isNew = store.collect(artworkId, pack.museumId, via);
    _collected = store.collectedIds();
    notifyListeners();
    return isNew;
  }

  void toggleFavorite(String id) {
    store.toggleFavorite(id);
    notifyListeners();
  }
}
