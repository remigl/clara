// Cœur de Clara : packs musées (SQLite + vecteurs), catalogue téléchargeable, modèle DINOv2
// (ONNX Runtime), collection de l'utilisateur. Même logique que la référence Python (pipeline/clara_p0).
//
// Réseau : uniquement pour télécharger le catalogue et les packs, à la demande de l'utilisateur.
// La reconnaissance, la collection et les fiches fonctionnent entièrement hors ligne.
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqlite3/sqlite3.dart';

const int inputSize = 224;
const String releasesApi = 'https://api.github.com/repos/remigl/clara/releases?per_page=10';
const String catalogUrl = 'https://github.com/remigl/clara/releases/latest/download/catalog.json';
const String packBaseUrl = 'https://github.com/remigl/clara/releases/latest/download/';
const String embeddedPackAsset = 'assets/packs/dk-smk.zip';

// ---------------------------------------------------------------------------- pays

const Map<String, String> countryNames = {
  'AT': 'Autriche', 'BE': 'Belgique', 'CH': 'Suisse', 'CZ': 'Tchéquie', 'DE': 'Allemagne', 'DK': 'Danemark',
  'ES': 'Espagne', 'FI': 'Finlande', 'FR': 'France', 'GB': 'Royaume-Uni', 'GR': 'Grèce', 'HU': 'Hongrie',
  'IE': 'Irlande', 'IT': 'Italie', 'LU': 'Luxembourg', 'NL': 'Pays-Bas', 'NO': 'Norvège', 'PL': 'Pologne',
  'PT': 'Portugal', 'RO': 'Roumanie', 'SE': 'Suède', 'UK': 'Royaume-Uni',
};

String countryFlag(String iso) {
  final code = iso.toUpperCase() == 'UK' ? 'GB' : iso.toUpperCase();
  if (code.length != 2) return '🏛';
  return String.fromCharCodes(code.codeUnits.map((c) => 0x1F1E6 + c - 65));
}

String countryLabel(String iso, [String? name]) =>
    (name != null && name.isNotEmpty) ? name : (countryNames[iso.toUpperCase()] ?? iso);

/// Clé de tri des noms en français (insensible aux accents et à la casse).
String sortKey(String s) {
  const from = 'àâäáãåçéèêëíìîïñóòôöõúùûüýÿœæ';
  const to = 'aaaaaaceeeeiiiinooooouuuuyyoa';
  final b = StringBuffer();
  for (final ch in s.toLowerCase().split('')) {
    final i = from.indexOf(ch);
    b.write(i >= 0 ? to[i] : ch);
  }
  return b.toString();
}

// ---------------------------------------------------------------------------- œuvres

class Artwork {
  Artwork.fromRow(Row r, this.pack)
      : id = r['id'] as String,
        museumId = r['museum_id'] as String,
        title = (r['title'] as String?) ?? 'Sans titre',
        artist = r['artist_display'] as String?,
        artistLife = r['artist_life'] as String?,
        artistNationality = r['artist_nationality'] as String?,
        date = r['date_display'] as String?,
        medium = r['medium'] as String?,
        description = r['description'] as String?,
        descriptionCredit = r['description_credit'] as String?,
        facts = _parseFacts(r['facts_json'] as String?),
        sections = _parseSections(r['sections_json'] as String?),
        artistId = r['artist_id'] as String?,
        room = r['room'] as String?,
        status = (r['exhibition_status'] as String?) ?? 'unknown',
        inventory = r['inventory_number'] as String?,
        heightCm = (r['height_cm'] as num?)?.toDouble(),
        widthCm = (r['width_cm'] as num?)?.toDouble(),
        sourceUrl = r['source_url'] as String?,
        thumb = r['local_path'] as String?,
        credit = r['credit_line'] as String?;

  final MuseumPack pack;
  final String id;
  final String museumId;
  final String title;
  final String? artist;
  final String? artistLife;
  final String? artistNationality;
  final String? date;
  final String? medium;
  final String? description;
  final String? descriptionCredit;
  final List<(String, String)> facts;
  final List<(String, List<(String, String)>)> sections;
  final String? artistId;
  final String? room;
  final String status;
  final String? inventory;
  final double? heightCm;
  final double? widthCm;
  final String? sourceUrl;
  final String? thumb;
  final String? credit;

  static List<(String, String)> _parseFacts(String? json) {
    if (json == null || json.isEmpty) return const [];
    try {
      return [
        for (final f in jsonDecode(json) as List)
          if (f is List && f.length == 2) ('${f[0]}', '${f[1]}')
      ];
    } catch (_) {
      return const [];
    }
  }

  static List<(String, List<(String, String)>)> _parseSections(String? json) {
    if (json == null || json.isEmpty) return const [];
    try {
      return [
        for (final sec in jsonDecode(json) as List)
          if (sec is List && sec.length == 2 && sec[1] is List)
            (
              '${sec[0]}',
              [
                for (final it in sec[1] as List)
                  if (it is List && it.length == 2) ('${it[0]}', '${it[1]}')
              ]
            )
      ];
    } catch (_) {
      return const [];
    }
  }

  File? get thumbFile => thumb == null ? null : File('${pack.dir.path}/$thumb');

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
        return '';
    }
  }

  String get artistLine {
    final extra = [artistLife, artistNationality].whereType<String>().where((s) => s.isNotEmpty).join(', ');
    final a = artist ?? 'Artiste inconnu';
    return extra.isEmpty ? a : '$a ($extra)';
  }
}

/// L'artiste, croisé depuis Wikidata et Wikipédia.
class ArtistInfo {
  ArtistInfo.fromRow(Row r)
      : name = (r['name'] as String?) ?? 'Artiste inconnu',
        birthYear = r['birth_year'] as int?,
        deathYear = r['death_year'] as int?,
        nationality = r['nationality'] as String?,
        birthPlace = r['birth_place'] as String?,
        deathPlace = r['death_place'] as String?,
        occupations = r['occupations'] as String?,
        movements = r['movements'] as String?,
        teachers = r['teachers'] as String?,
        students = r['students'] as String?,
        bio = r['bio'] as String?,
        bioCredit = r['bio_credit'] as String?;

  final String name;
  final int? birthYear, deathYear;
  final String? nationality, birthPlace, deathPlace, occupations, movements, teachers, students, bio, bioCredit;

  String? get lifeLine {
    String part(int? y, String? place, String verb) =>
        [if (y != null) '$verb en $y', if (place != null && place.isNotEmpty) 'à $place'].join(' ');
    final born = part(birthYear, birthPlace, 'né');
    final died = part(deathYear, deathPlace, 'mort');
    final l = [born, died].where((x) => x.isNotEmpty).join(', ');
    return l.isEmpty ? null : l[0].toUpperCase() + l.substring(1);
  }

  bool get hasDetails =>
      [bio, birthPlace, deathPlace, movements, teachers, students, occupations].any((x) => x != null && x.isNotEmpty);
}

class GlossaryEntry {
  GlossaryEntry(this.term, this.text, this.credit);
  final String term;
  final String text;
  final String? credit;
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
  MuseumPack._(this.dir, this.manifest, this.db, this.dim, this.vectors, this.rowArtwork, this.museum,
      this._countable);

  final Directory dir;
  final Map<String, dynamic> manifest;
  final Database db;
  final int dim;
  final Float32List vectors; // lignes déquantifiées, normalisées L2
  final List<String> rowArtwork;
  final Map<String, Object?> museum;
  final Set<String> _countable;

  String get id => manifest['id'] as String;
  String get modelId => manifest['model_id'] as String;
  int get version => manifest['version'] as int;
  String? get builtAt => manifest['built_at'] as String?;
  Map<String, dynamic> get _m => (manifest['museum'] as Map?)?.cast<String, dynamic>() ?? const {};
  String get name => (museum['name_local'] as String?) ?? (museum['name'] as String? ?? id);
  String get city => (_m['city'] as String?) ?? (museum['city'] as String?) ?? '';
  String get country => (museum['country'] as String?) ?? '';
  String get countryName => countryLabel(country, _m['country_name'] as String?);
  int get numberOfWorks => (museum['number_of_works'] as int?) ?? 0;
  int get dataQuality => (museum['data_quality'] as int?) ?? 0;
  bool get knowsDisplay => dataQuality >= 2;
  bool countsForProgress(String artworkId) => _countable.contains(artworkId);

  static Future<MuseumPack> loadDir(Directory dir) async {
    final manifest = jsonDecode(await File('${dir.path}/manifest.json').readAsString()) as Map<String, dynamic>;
    final db = sqlite3.open('${dir.path}/catalog.db', mode: OpenMode.readOnly);
    final dim = manifest['embedding_dim'] as int;
    final qi = Int8List.fromList(await File('${dir.path}/vectors.bin').readAsBytes());
    final sb = await File('${dir.path}/scales.bin').readAsBytes();
    final sd = ByteData.sublistView(sb);
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
      throw StateError('Pack ${manifest['id']} incohérent : ${rowArtwork.length} lignes, $n vecteurs');
    }
    final museum = Map<String, Object?>.from(db.select('SELECT * FROM museum LIMIT 1').first);
    final countable = {
      for (final r in db.select('SELECT id FROM artwork WHERE counts_for_progress = 1')) r['id'] as String
    };
    return MuseumPack._(dir, manifest, db, dim, vectors, rowArtwork, museum, countable);
  }

  void close() => db.dispose();

  Artwork? artwork(String id) {
    final r = db.select('$_artworkSelect WHERE a.id = ?', [id]);
    return r.isEmpty ? null : Artwork.fromRow(r.first, this);
  }

  List<Artwork> all() => [
        for (final r in db.select('$_artworkSelect ORDER BY a.room IS NULL, a.room, a.title')) Artwork.fromRow(r, this)
      ];

  ArtistInfo? artist(String? id) {
    if (id == null) return null;
    final r = db.select('SELECT * FROM artist WHERE id = ?', [id]);
    return r.isEmpty ? null : ArtistInfo.fromRow(r.first);
  }

  /// Notions expliquées (mouvements, genres) liées à une œuvre. Les packs anciens n'ont pas ces tables.
  List<GlossaryEntry> glossaryFor(String artworkId) {
    try {
      return [
        for (final r in db.select(
            'SELECT g.term, g.text, g.credit FROM artwork_glossary ag JOIN glossary g ON g.term = ag.term '
            'WHERE ag.artwork_id = ?',
            [artworkId]))
          GlossaryEntry(r['term'] as String, r['text'] as String, r['credit'] as String?)
      ];
    } on SqliteException {
      return const [];
    }
  }

  /// Autres œuvres du même artiste dans ce musée (celles avec image d'abord).
  List<Artwork> otherWorksBy(String? artistId, String excludeId, {int limit = 20}) {
    if (artistId == null) return const [];
    return [
      for (final r in db.select(
          '$_artworkSelect WHERE a.artist_id = ? AND a.id != ? ORDER BY i.local_path IS NULL, a.date_start LIMIT ?',
          [artistId, excludeId, limit]))
        Artwork.fromRow(r, this)
    ];
  }

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
        'LEFT JOIN artwork_image i ON i.artwork_id = a.id WHERE artwork_fts MATCH ? LIMIT 200',
        [terms]);
    return [for (final row in r) Artwork.fromRow(row, this)];
  }

  /// Plus proches œuvres : max sur les vues de la requête et les vues de référence d'une même œuvre.
  List<(String, double)> nearestIds(List<Float32List> queries, {int k = 3}) {
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
    return [for (final e in sorted.take(k)) (e.key, e.value)];
  }
}

// ---------------------------------------------------------------------------- catalogue distant

class RemotePack {
  RemotePack(this.json);
  final Map<String, dynamic> json;
  String get id => json['id'] as String;
  String get name => (json['name'] as String?) ?? id;
  String get city => (json['city'] as String?) ?? '';
  String get country => (json['country'] as String?) ?? '';
  String get countryName => countryLabel(country, json['country_name'] as String?);
  int get version => (json['version'] as num?)?.toInt() ?? 1;
  int get works => (json['works'] as num?)?.toInt() ?? 0;
  int get recognizable => (json['recognizable'] as num?)?.toInt() ?? 0;
  int get bytes => (json['bytes'] as num?)?.toInt() ?? 0;
  String get file => json['file'] as String;
  String get sha256 => (json['sha256'] as String?) ?? '';
  String get modelId => (json['model_id'] as String?) ?? '';
  int get dataQuality => (json['data_quality'] as num?)?.toInt() ?? 0;
  String get url => '${(json['base_url'] as String?) ?? packBaseUrl}$file';
  String? get builtAt => json['built_at'] as String?;
}

String formatBytes(int b) {
  if (b >= 1e9) return '${(b / 1e9).toStringAsFixed(1)} Go';
  if (b >= 1e6) return '${(b / 1e6).toStringAsFixed(0)} Mo';
  return '${(b / 1e3).toStringAsFixed(0)} Ko';
}

class PackStore {
  PackStore(this.root);
  final Directory root;

  static Future<PackStore> open() async {
    final dir = await getApplicationSupportDirectory();
    final root = Directory('${dir.path}/packs')..createSync(recursive: true);
    return PackStore(root);
  }

  File get catalogCache => File('${root.path}/catalog.json');

  Directory dirOf(String id) => Directory('${root.path}/$id');

  List<Directory> installedDirs() => [
        for (final e in root.listSync())
          if (e is Directory && !e.path.contains('.tmp-') && File('${e.path}/manifest.json').existsSync()) e
      ];

  Map<String, dynamic>? installedManifest(String id) {
    final f = File('${dirOf(id).path}/manifest.json');
    if (!f.existsSync()) return null;
    try {
      return jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
    } catch (_) {
      return null;
    }
  }

  /// Installe un pack depuis les octets d'un zip (remplace l'éventuelle version précédente).
  Future<String> installZipBytes(Uint8List bytes) async {
    final archive = ZipDecoder().decodeBytes(bytes);
    final manifestFile = archive.files.firstWhere((f) => f.name == 'manifest.json',
        orElse: () => throw const FormatException('manifest.json absent du pack'));
    final manifest = jsonDecode(utf8.decode(manifestFile.content as List<int>)) as Map<String, dynamic>;
    final id = manifest['id'] as String;
    final tmp = Directory('${root.path}/.tmp-$id');
    if (tmp.existsSync()) tmp.deleteSync(recursive: true);
    tmp.createSync(recursive: true);
    for (final f in archive.files) {
      if (!f.isFile) continue;
      final name = f.name.replaceAll('\\', '/');
      if (name.contains('..')) continue; // sécurité : pas de sortie du dossier
      final out = File('${tmp.path}/$name');
      out.parent.createSync(recursive: true);
      out.writeAsBytesSync(f.content as List<int>);
    }
    final dest = dirOf(id);
    if (dest.existsSync()) dest.deleteSync(recursive: true);
    tmp.renameSync(dest.path);
    return id;
  }

  /// Pack embarqué dans l'app (SMK) : installé au premier lancement, et à chaque nouvelle version de l'app.
  Future<void> installEmbedded() async {
    ByteData data;
    try {
      data = await rootBundle.load(embeddedPackAsset);
    } catch (_) {
      return; // app construite sans pack embarqué
    }
    final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    final archive = ZipDecoder().decodeBytes(bytes);
    final mf = archive.files.firstWhere((f) => f.name == 'manifest.json');
    final m = jsonDecode(utf8.decode(mf.content as List<int>)) as Map<String, dynamic>;
    final current = installedManifest(m['id'] as String);
    if (current != null &&
        current['model_id'] == m['model_id'] &&
        ((current['built_at'] as String?) ?? '').compareTo((m['built_at'] as String?) ?? '') >= 0) {
      return;
    }
    await installZipBytes(bytes);
  }

  void delete(String id) {
    final d = dirOf(id);
    if (d.existsSync()) d.deleteSync(recursive: true);
  }

  Future<List<RemotePack>?> cachedCatalog() async {
    if (!catalogCache.existsSync()) return null;
    try {
      return _parseCatalog(await catalogCache.readAsString());
    } catch (_) {
      return null;
    }
  }

  List<RemotePack> _parseCatalog(String body) {
    final j = jsonDecode(body) as Map<String, dynamic>;
    final model = j['model_id'] as String?;
    final base = j['base_url'] as String?;
    return [
      for (final p in (j['packs'] as List? ?? const []))
        RemotePack({'model_id': model, 'base_url': base, ...(p as Map).cast<String, dynamic>()})
    ];
  }

  Future<String> _get(HttpClient client, String url, {bool json = false}) async {
    final req = await client.getUrl(Uri.parse(url));
    req.headers.set(HttpHeaders.userAgentHeader, 'Clara-app');
    if (json) req.headers.set(HttpHeaders.acceptHeader, 'application/vnd.github+json');
    final res = await req.close();
    if (res.statusCode != 200) {
      await res.drain<void>();
      throw HttpException('HTTP ${res.statusCode}', uri: Uri.parse(url));
    }
    return res.transform(utf8.decoder).join();
  }

  /// Catalogue de la release la plus récente qui en contient un : pendant la construction des packs
  /// d'une nouvelle version (plusieurs heures), l'app continue d'utiliser ceux de la version précédente.
  Future<List<RemotePack>> fetchCatalog() async {
    final client = HttpClient()..connectionTimeout = const Duration(seconds: 15);
    try {
      String body;
      try {
        final releases = jsonDecode(await _get(client, releasesApi, json: true)) as List;
        String? found;
        for (final r in releases.cast<Map<String, dynamic>>()) {
          if (r['draft'] == true || r['prerelease'] == true) continue;
          final assets = (r['assets'] as List? ?? const []).cast<Map<String, dynamic>>();
          final cat = assets.where((a) => a['name'] == 'catalog.json').firstOrNull;
          if (cat == null) continue;
          final j = jsonDecode(await _get(client, cat['browser_download_url'] as String)) as Map<String, dynamic>;
          j['base_url'] = 'https://github.com/remigl/clara/releases/download/${r['tag_name']}/';
          j['release'] = r['tag_name'];
          found = jsonEncode(j);
          break;
        }
        if (found == null) throw const HttpException('aucune release ne contient encore de catalogue');
        body = found;
      } on HttpException catch (e) {
        if (e.message.contains('aucune release')) rethrow;
        body = await _get(client, catalogUrl); // API GitHub limitée (60 requêtes/heure) : dernière release
      }
      final packs = _parseCatalog(body);
      await catalogCache.writeAsString(body);
      return packs;
    } finally {
      client.close();
    }
  }

  /// Télécharge, vérifie (SHA-256) et installe un pack. onProgress(reçus, total).
  Future<String> download(RemotePack p, void Function(int, int) onProgress) async {
    final client = HttpClient()..connectionTimeout = const Duration(seconds: 20);
    final tmp = File('${root.path}/.download-${p.id}.zip');
    try {
      final req = await client.getUrl(Uri.parse(p.url));
      final res = await req.close();
      if (res.statusCode != 200) throw HttpException('téléchargement impossible (HTTP ${res.statusCode})');
      final total = res.contentLength > 0 ? res.contentLength : p.bytes;
      final sink = tmp.openWrite();
      final digestSink = _DigestSink();
      final hasher = sha256.startChunkedConversion(digestSink);
      var received = 0;
      await for (final chunk in res) {
        sink.add(chunk);
        hasher.add(chunk);
        received += chunk.length;
        onProgress(received, total);
      }
      await sink.close();
      hasher.close();
      if (p.sha256.isNotEmpty && digestSink.value.toString() != p.sha256) {
        throw const FormatException('fichier corrompu (empreinte SHA-256 différente), réessaie');
      }
      return await installZipBytes(await tmp.readAsBytes());
    } finally {
      client.close();
      if (tmp.existsSync()) tmp.deleteSync();
    }
  }
}

class _DigestSink implements Sink<Digest> {
  late Digest value;
  @override
  void add(Digest data) => value = data;
  @override
  void close() {}
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
        if (v is List && v.isNotEmpty && v[0] is List && (v[0] as List).isNotEmpty && (v[0] as List)[0] is List) {
          final cls = ((v[0] as List)[0] as List).map((e) => (e as num).toDouble()).toList();
          return l2normalize(cls);
        }
      }
      final v = outputs.first?.value as List;
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
    db.execute('''CREATE TABLE IF NOT EXISTS setting (key TEXT PRIMARY KEY, value TEXT)''');
    return UserStore._(db);
  }

  static String _now() => DateTime.now().toUtc().toIso8601String();

  String? getSetting(String key) {
    final r = db.select('SELECT value FROM setting WHERE key = ?', [key]);
    return r.isEmpty ? null : r.first['value'] as String?;
  }

  void setSetting(String key, String? value) {
    if (value == null) {
      db.execute('DELETE FROM setting WHERE key = ?', [key]);
    } else {
      db.execute('INSERT INTO setting VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value', [key, value]);
    }
  }

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

  List<(String, String)> collected() => [
        for (final r in db.select('SELECT artwork_id, museum_id FROM collected ORDER BY last_seen_at DESC'))
          (r['artwork_id'] as String, r['museum_id'] as String)
      ];

  bool isFavorite(String id) => db.select('SELECT 1 FROM favorite WHERE artwork_id = ?', [id]).isNotEmpty;

  void toggleFavorite(String id) {
    if (isFavorite(id)) {
      db.execute('DELETE FROM favorite WHERE artwork_id = ?', [id]);
    } else {
      db.execute('INSERT INTO favorite VALUES (?, ?)', [id, _now()]);
    }
  }

  void logScan(String? museumId, List<Candidate> c, String decision, {String? chosen}) {
    db.execute(
        'INSERT INTO scan (created_at, museum_id, artwork_id, score, decision, candidates_json) VALUES (?,?,?,?,?,?)', [
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

const String everywhere = '*';

class AppState extends ChangeNotifier {
  AppState._(this.store, this.packStore, this.embedder);

  final UserStore store;
  final PackStore packStore;
  final Embedder embedder;
  final Map<String, MuseumPack> packs = {};
  List<RemotePack>? catalog;
  String? catalogError;
  bool catalogLoading = false;
  final Map<String, double> downloads = {}; // id -> progression 0..1
  late List<(String, String)> _collected;
  late Set<String> _collectedIds;

  static Future<AppState> load() async {
    final store = await UserStore.open();
    final packStore = await PackStore.open();
    await packStore.installEmbedded();
    final embedder = await Embedder.load();
    final s = AppState._(store, packStore, embedder);
    for (final d in packStore.installedDirs()) {
      try {
        final p = await MuseumPack.loadDir(d);
        s.packs[p.id] = p;
      } catch (e) {
        debugPrint('pack ${d.path} illisible : $e');
      }
    }
    s._refreshCollected();
    s.catalog = await packStore.cachedCatalog();
    unawaited(s.refreshCatalog());
    return s;
  }

  /// model_id commun (pack embarqué) : les packs du catalogue doivent avoir le même.
  String? get modelId => packs.values.isEmpty ? null : packs.values.first.modelId;

  void _refreshCollected() {
    _collected = store.collected();
    _collectedIds = {for (final c in _collected) c.$1};
  }

  // --- musée courant

  String get currentId {
    final v = store.getSetting('current_museum');
    if (v == everywhere) return everywhere;
    if (v != null && packs.containsKey(v)) return v;
    return packs.isEmpty ? everywhere : sortedPacks().first.id;
  }

  MuseumPack? get current => currentId == everywhere ? null : packs[currentId];

  void setCurrent(String id) {
    store.setSetting('current_museum', id);
    notifyListeners();
  }

  List<MuseumPack> sortedPacks() => packs.values.toList()
    ..sort((a, b) {
      final c = sortKey(a.countryName).compareTo(sortKey(b.countryName));
      if (c != 0) return c;
      final v = sortKey(a.city).compareTo(sortKey(b.city));
      return v != 0 ? v : sortKey(a.name).compareTo(sortKey(b.name));
    });

  // --- collection

  bool isCollected(String id) => _collectedIds.contains(id);

  int collectedCount(MuseumPack p) =>
      _collected.where((c) => c.$2 == p.id && p.countsForProgress(c.$1)).length;

  double progress(MuseumPack p) => p.numberOfWorks == 0 ? 0 : collectedCount(p) / p.numberOfWorks;

  /// Œuvres collectionnées d'un musée (seulement si son pack est installé).
  List<Artwork> collectedIn(MuseumPack p) => [
        for (final c in _collected)
          if (c.$2 == p.id)
            if (p.artwork(c.$1) case final a?) a
      ];

  /// Nombre d'œuvres collectionnées par musée, pack installé ou non.
  Map<String, int> collectedByMuseum() {
    final m = <String, int>{};
    for (final c in _collected) {
      m[c.$2] = (m[c.$2] ?? 0) + 1;
    }
    return m;
  }

  bool add(Artwork a, {String via = 'scan'}) {
    final isNew = store.collect(a.id, a.museumId, via);
    _refreshCollected();
    notifyListeners();
    return isNew;
  }

  void toggleFavorite(String id) {
    store.toggleFavorite(id);
    notifyListeners();
  }

  // --- reconnaissance

  Future<List<Candidate>> recognize(Uint8List jpeg) async {
    final views = await compute(prepareViews, jpeg);
    final q = <Float32List>[];
    for (final v in views) {
      q.add(await embedder.embed(v));
    }
    final targets = current != null ? [current!] : packs.values.toList();
    final all = <(MuseumPack, String, double)>[];
    for (final p in targets) {
      if (q.isNotEmpty && q.first.length != p.dim) continue; // pack d'un autre modèle
      for (final (id, s) in p.nearestIds(q, k: 3)) {
        all.add((p, id, s));
      }
    }
    all.sort((a, b) => b.$3.compareTo(a.$3));
    return [
      for (final (p, id, s) in all.take(3))
        if (p.artwork(id) case final a?) Candidate(a, s)
    ];
  }

  // --- catalogue et téléchargements

  Future<void> refreshCatalog() async {
    if (catalogLoading) return;
    catalogLoading = true;
    notifyListeners();
    try {
      catalog = await packStore.fetchCatalog();
      catalogError = null;
    } catch (e) {
      catalogError = e is SocketException
          ? 'pas de connexion Internet'
          : e is HttpException
              ? e.message
              : '$e';
    } finally {
      catalogLoading = false;
      notifyListeners();
    }
  }

  bool isCompatible(RemotePack r) => modelId == null || r.modelId.isEmpty || r.modelId == modelId;

  /// Mise à jour proposée seulement si le pack a vraiment été reconstruit (les packs inchangés sont repris
  /// d'une release à l'autre avec la même date de construction) : pas de nouveau téléchargement inutile.
  bool hasUpdate(RemotePack r) {
    final p = packs[r.id];
    if (p == null || !isCompatible(r)) return false;
    if (r.version > p.version) return true;
    final a = r.builtAt, b = p.builtAt;
    return a != null && b != null && a.compareTo(b) > 0;
  }

  Future<void> download(RemotePack r) async {
    if (downloads.containsKey(r.id)) return;
    downloads[r.id] = 0;
    notifyListeners();
    var last = DateTime.now();
    try {
      await packStore.download(r, (got, total) {
        if (DateTime.now().difference(last).inMilliseconds > 200) {
          last = DateTime.now();
          downloads[r.id] = total > 0 ? got / total : 0;
          notifyListeners();
        }
      });
      packs.remove(r.id)?.close();
      packs[r.id] = await MuseumPack.loadDir(packStore.dirOf(r.id));
    } finally {
      downloads.remove(r.id);
      notifyListeners();
    }
  }

  void remove(String id) {
    packs.remove(id)?.close();
    packStore.delete(id);
    if (store.getSetting('current_museum') == id) store.setSetting('current_museum', null);
    notifyListeners();
  }
}
