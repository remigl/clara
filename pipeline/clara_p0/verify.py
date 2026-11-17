"""Vérification géométrique (reranking) : points d'intérêt + RANSAC homographie.

P0 utilise SIFT (OpenCV, brevet expiré, libre) avec RootSIFT. Sur l'appareil, XFeat (Apache 2.0)
le remplacera pour la vitesse ; l'interface (nombre d'inliers) reste la même.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .imageio import resize_max


@dataclass
class Keypoints:
    xy: np.ndarray    # (n, 2) float32, coordonnées dans l'image réduite
    desc: np.ndarray  # (n, 128) uint8 (SIFT)

    def to_bytes(self) -> bytes:
        n = len(self.xy)
        return struct.pack("<I", n) + self.xy.astype(np.float32).tobytes() + self.desc.astype(np.uint8).tobytes()

    @classmethod
    def from_bytes(cls, b: bytes) -> "Keypoints":
        (n,) = struct.unpack_from("<I", b, 0)
        xy = np.frombuffer(b, np.float32, n * 2, 4).reshape(n, 2)
        desc = np.frombuffer(b, np.uint8, n * 128, 4 + n * 8).reshape(n, 128)
        return cls(xy, desc)


class GeometricVerifier:
    def __init__(self, max_side: int = 640, n_features: int = 500, ratio: float = 0.8,
                 ransac_px: float = 6.0):
        self.max_side = max_side
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.ratio = ratio
        self.ransac_px = ransac_px
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def extract(self, img: np.ndarray) -> Keypoints:
        g = cv2.cvtColor(resize_max(img, self.max_side), cv2.COLOR_BGR2GRAY)
        kps, desc = self.sift.detectAndCompute(g, None)
        if desc is None or len(kps) == 0:
            return Keypoints(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
        xy = np.array([k.pt for k in kps], np.float32)
        return Keypoints(xy, np.clip(desc, 0, 255).astype(np.uint8))

    @staticmethod
    def _root(desc: np.ndarray) -> np.ndarray:
        d = desc.astype(np.float32)
        d /= np.maximum(d.sum(1, keepdims=True), 1e-6)
        return np.sqrt(d)

    def inliers(self, a: Keypoints, b: Keypoints) -> int:
        if len(a.xy) < 8 or len(b.xy) < 8:
            return 0
        knn = self.matcher.knnMatch(self._root(a.desc), self._root(b.desc), k=2)
        good = [m for m, *rest in (p for p in knn if len(p) == 2) if m.distance < self.ratio * rest[0].distance]
        if len(good) < 8:
            return 0
        src = a.xy[[m.queryIdx for m in good]]
        dst = b.xy[[m.trainIdx for m in good]]
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        if h is None or mask is None:
            return 0
        # homographie dégénérée (retournement, écrasement) -> rejet
        det = np.linalg.det(h[:2, :2])
        if det <= 0.02 or det > 50:
            return 0
        return int(mask.sum())


class LazyKeypoints:
    """Keypoints calculés à la demande depuis l'image de référence (grandes bases d'évaluation)."""

    def __init__(self, path_of, verifier: "GeometricVerifier", cache_size: int = 4096):
        from .imageio import read_image
        self._read = read_image
        self.path_of = path_of  # artwork_id -> chemin
        self.verifier = verifier
        self.cache: dict[str, Keypoints] = {}
        self.cache_size = cache_size

    def get(self, k):
        if k in self.cache:
            return self.cache[k]
        p = self.path_of(k)
        if p is None:
            return None
        kp = self.verifier.extract(self._read(p))
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[k] = kp
        return kp


class KeypointStore:
    """keypoints.bin + table d'offsets (même format que le pack de l'app)."""

    def __init__(self):
        self.items: dict[str, Keypoints] = {}

    def __setitem__(self, k, v):
        self.items[k] = v

    def get(self, k):
        return self.items.get(k)

    def save(self, path: str | Path) -> dict[str, tuple[int, int]]:
        offsets, pos = {}, 0
        with open(path, "wb") as f:
            for k, kp in self.items.items():
                b = kp.to_bytes()
                f.write(b)
                offsets[k] = (pos, len(b))
                pos += len(b)
        return offsets

    @classmethod
    def load(cls, path: str | Path, offsets: dict[str, tuple[int, int]]) -> "KeypointStore":
        s = cls()
        data = Path(path).read_bytes()
        for k, (o, n) in offsets.items():
            s.items[k] = Keypoints.from_bytes(data[o:o + n])
        return s
