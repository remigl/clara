"""Encodeurs d'images -> embeddings normalisés L2.

- OnnxDinoEmbedder : DINOv2 exporté en ONNX (ex. onnx-community/dinov2-small, Apache 2.0).
  Descripteur = [CLS ; moyenne GeM des patchs], puis projection optionnelle (PCA-whitening) et L2.
- ThumbEmbedder : descripteur trivial (vignette Lab 16x16). Sert uniquement aux tests hors ligne
  et de plancher de comparaison : un vrai modèle doit faire nettement mieux.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def l2n(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


class Projection:
    """PCA-whitening appris une fois (sur un grand ensemble), partagé par tous les packs d'un même model_id."""

    def __init__(self, mean: np.ndarray, matrix: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.matrix = matrix.astype(np.float32)  # (d_in, d_out)
        self.fingerprint = hashlib.sha256(self.mean.tobytes() + self.matrix.tobytes()).hexdigest()[:6]

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return l2n((x - self.mean) @ self.matrix)

    @classmethod
    def fit(cls, x: np.ndarray, dim: int = 256, shrinkage: float = 0.5) -> "Projection":
        x = x.astype(np.float64)
        mean = x.mean(0)
        xc = x - mean
        cov = xc.T @ xc / max(len(x) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1][:dim]
        evals, evecs = evals[order], evecs[:, order]
        # shrinkage : 0 = PCA simple, 1 = whitening complet
        matrix = evecs / np.power(np.maximum(evals, 1e-9), 0.5 * shrinkage)
        return cls(mean, matrix)

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, matrix=self.matrix)

    @classmethod
    def load(cls, path: str | Path) -> "Projection":
        d = np.load(path)
        return cls(d["mean"], d["matrix"])


class Embedder:
    model_id: str
    dim: int

    def embed(self, images: list[np.ndarray]) -> np.ndarray:  # (n, dim) float32, L2
        raise NotImplementedError


class ThumbEmbedder(Embedder):
    def __init__(self, size: int = 16):
        self.size = size
        self.dim = self.raw_dim = size * size * 3
        self.model_id = self.raw_id = self.legacy_id = f"thumb-lab-{size}"
        self.projection = None

    # même interface que OnnxDinoEmbedder (tests)
    def _raw(self, images):
        return self.embed(images)

    def embed_raw(self, images):
        return self.embed(images)

    def project(self, x):
        return l2n(x).astype(np.float32)

    def from_legacy(self, legacy_vectors):
        return None

    def embed(self, images):
        out = []
        for img in images:
            lab = cv2.cvtColor(cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2LAB).astype(np.float32)
            v = lab.reshape(-1)
            out.append(v - v.mean())
        return l2n(np.stack(out)).astype(np.float32)


POOLINGS = ("cls", "gem", "cls+gem")


class OnnxDinoEmbedder(Embedder):
    """pooling : 'cls' (défaut, meilleur au comparatif Met), 'gem' ou 'cls+gem' (réglage initial de P0).

    raw_id identifie l'empreinte AVANT projection : c'est la clé des caches d'embeddings, si bien qu'on
    peut changer de projection sans réencoder. model_id = raw_id + projection : c'est ce qu'enregistre un pack.
    """

    def __init__(self, model_path: str | Path, size: int = 224, projection: Projection | None = None,
                 threads: int = 0, gem_p: float = 3.0, pooling: str = "cls"):
        if pooling not in POOLINGS:
            raise ValueError(f"pooling inconnu : {pooling}")
        self.pooling = pooling
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.size = size
        self.gem_p = gem_p
        self.projection = projection
        probe = self._raw([np.zeros((size, size, 3), np.uint8)])
        self.raw_dim = probe.shape[1]
        self.dim = projection.dim if projection else self.raw_dim
        h = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:8]
        # 'cls+gem' garde l'identifiant historique pour réutiliser les caches existants
        self.legacy_id = f"dinov2-onnx-{h}-{size}"
        self.raw_id = self.legacy_id if pooling == "cls+gem" else f"{self.legacy_id}-{pooling}"
        self.model_id = self.raw_id + (f"-p{self.dim}-{projection.fingerprint}" if projection else "")

    def _prep(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        s = self.size / min(h, w)
        img = cv2.resize(img, (max(self.size, round(w * s)), max(self.size, round(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        y0, x0 = (h - self.size) // 2, (w - self.size) // 2
        img = img[y0:y0 + self.size, x0:x0 + self.size]
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1)

    def _raw(self, images: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([self._prep(i) for i in images]).astype(np.float32)
        outs = dict(zip(self.output_names, self.sess.run(None, {self.input_name: batch})))
        tokens = outs.get("last_hidden_state")
        if tokens is None:  # export maison : premier tenseur 3D, sinon 2D
            tokens = next((v for v in outs.values() if v.ndim == 3), None)
        if tokens is None:
            return l2n(next(iter(outs.values())).astype(np.float32))
        cls = l2n(tokens[:, 0])
        if self.pooling == "cls":
            return cls.astype(np.float32)
        patches = np.maximum(tokens[:, 1:], 1e-6)  # GeM sur valeurs positives
        gem = l2n(np.power(np.mean(np.power(patches, self.gem_p), axis=1), 1.0 / self.gem_p))
        if self.pooling == "gem":
            return gem.astype(np.float32)
        return np.concatenate([cls, gem], axis=1).astype(np.float32)

    def embed_raw(self, images, batch_size: int = 16) -> np.ndarray:
        """Empreinte avant projection, normalisée L2."""
        out = [self._raw(images[i:i + batch_size]) for i in range(0, len(images), batch_size)]
        x = np.concatenate(out) if out else np.zeros((0, self.raw_dim), np.float32)
        return l2n(x).astype(np.float32)

    def project(self, x: np.ndarray) -> np.ndarray:
        return (self.projection(x) if self.projection else l2n(x)).astype(np.float32)

    def embed(self, images, batch_size: int = 16):
        return self.project(self.embed_raw(images, batch_size))

    def from_legacy(self, legacy_vectors: np.ndarray) -> np.ndarray | None:
        """Retrouve l'empreinte 'cls' ou 'gem' à partir d'un cache 'cls+gem' (moitiés normalisées)."""
        h = legacy_vectors.shape[1] // 2
        if self.pooling == "cls":
            return l2n(legacy_vectors[:, :h]).astype(np.float32)
        if self.pooling == "gem":
            return l2n(legacy_vectors[:, h:]).astype(np.float32)
        return None


def make_embedder(model: str | None, projection: str | None = None, size: int = 224,
                  pooling: str = "cls") -> Embedder:
    """model=None ou 'thumb' -> ThumbEmbedder ; sinon chemin d'un .onnx. projection=None ou 'none' : aucune."""
    if not model or model == "thumb":
        return ThumbEmbedder()
    proj = Projection.load(projection) if projection and projection != "none" else None
    return OnnxDinoEmbedder(model, size=size, projection=proj, pooling=pooling)
