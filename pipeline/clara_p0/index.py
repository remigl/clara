"""Index vectoriel exact int8 (brute force), même format que celui lu par l'app (vectors.bin).

Format sur disque :
  vectors.bin  int8[n_rows, dim], ligne par ligne
  scales.bin   float32[n_rows]   (v ≈ int8 * scale)
  rows.json    [[artwork_id, view_idx, kind], ...]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantification symétrique par vecteur : q = round(v / max|v| * 127)."""
    m = np.maximum(np.abs(x).max(axis=1), 1e-12)
    scale = (m / 127.0).astype(np.float32)
    q = np.clip(np.round(x / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


class FlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.q = np.zeros((0, dim), np.int8)
        self.scale = np.zeros((0,), np.float32)
        self.rows: list[tuple[str, int, str]] = []

    def __len__(self):
        return len(self.rows)

    @property
    def artwork_ids(self) -> list[str]:
        return sorted({r[0] for r in self.rows})

    def add(self, vectors: np.ndarray, rows: list[tuple[str, int, str]]):
        assert vectors.shape == (len(rows), self.dim)
        q, s = quantize(vectors.astype(np.float32))
        self.q = np.concatenate([self.q, q])
        self.scale = np.concatenate([self.scale, s])
        self.rows.extend(rows)
        self._group = None

    def _groups(self):
        if getattr(self, "_group", None) is None:
            ids = [r[0] for r in self.rows]
            uniq = sorted(set(ids))
            pos = {a: i for i, a in enumerate(uniq)}
            self._uniq = uniq
            self._group = np.array([pos[a] for a in ids], np.int64)
            self._qf = self.q.astype(np.float32) * self.scale[:, None]  # cache de déquantification
        return self._uniq, self._group

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Similarités cosinus de chaque requête (m, dim) avec chaque ligne -> (m, n_rows).

        La requête est elle aussi quantifiée int8 (comme sur l'appareil) ; le calcul est fait en float
        ici pour la portabilité numpy, ce qui est numériquement identique au produit int8 * échelles.
        """
        query = np.atleast_2d(query).astype(np.float32)
        qq, qs = quantize(query)
        self._groups()
        return (qq.astype(np.float32) * qs[:, None]) @ self._qf.T

    def search(self, query: np.ndarray, k: int = 5) -> list[list[tuple[str, float]]]:
        """Top-k œuvres par requête (max sur les vues d'une même œuvre)."""
        uniq, group = self._groups()
        s = self.scores(query)
        out = []
        for row in s:
            best = np.full(len(uniq), -np.inf, np.float32)
            np.maximum.at(best, group, row)
            k_ = min(k, len(uniq))
            top = np.argpartition(-best, k_ - 1)[:k_]
            top = top[np.argsort(-best[top])]
            out.append([(uniq[i], float(best[i])) for i in top])
        return out

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.q.tofile(d / "vectors.bin")
        self.scale.astype(np.float32).tofile(d / "scales.bin")
        (d / "rows.json").write_text(json.dumps({"dim": self.dim, "rows": self.rows}), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "FlatIndex":
        d = Path(directory)
        meta = json.loads((d / "rows.json").read_text(encoding="utf-8"))
        idx = cls(meta["dim"])
        idx.rows = [tuple(r) for r in meta["rows"]]
        idx.q = np.fromfile(d / "vectors.bin", dtype=np.int8).reshape(len(idx.rows), idx.dim)
        idx.scale = np.fromfile(d / "scales.bin", dtype=np.float32)
        return idx
