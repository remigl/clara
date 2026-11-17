import 'package:flutter/material.dart';

import 'core.dart';
import 'screens.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ClaraApp());
}

class ClaraApp extends StatelessWidget {
  const ClaraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Clara',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF8C5A2B),
        useMaterial3: true,
      ),
      home: const BootScreen(),
    );
  }
}

/// Charge le pack, le modèle et la collection (une seule fois, hors ligne).
class BootScreen extends StatefulWidget {
  const BootScreen({super.key});

  @override
  State<BootScreen> createState() => _BootScreenState();
}

class _BootScreenState extends State<BootScreen> {
  late final Future<AppState> _future = AppState.load();

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<AppState>(
      future: _future,
      builder: (context, snap) {
        if (snap.hasError) {
          return Scaffold(
            body: Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text('Impossible de démarrer Clara :\n${snap.error}', textAlign: TextAlign.center),
              ),
            ),
          );
        }
        if (!snap.hasData) {
          return const Scaffold(
            body: Center(
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                Text('Clara', style: TextStyle(fontSize: 40, fontWeight: FontWeight.w300, letterSpacing: 4)),
                SizedBox(height: 24),
                CircularProgressIndicator(),
              ]),
            ),
          );
        }
        return HomeShell(state: snap.data!);
      },
    );
  }
}
