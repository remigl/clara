"""Données synthétiques.

1. `procedural_painting` : fausses « peintures » (pour tester la chaîne sans aucune donnée externe).
2. `museum_photo` : transforme une image de référence en photo de musée plausible
   (cadre, mur, perspective, éclairage, reflet, flou, bruit, occlusion, JPEG).
   C'est le même générateur qui servira à entraîner le détecteur et à fine-tuner l'embedding (P2).
"""
from __future__ import annotations

import cv2
import numpy as np


def procedural_painting(rng: np.random.Generator, w: int = 600, h: int = 450,
                        base: np.ndarray | None = None) -> np.ndarray:
    """Composition aléatoire : fond dégradé, aplats, coups de pinceau, texture.
    Si `base` est fourni, produit une variante proche (même série, autre tableau)."""
    if base is not None:
        img = base.copy()
        for _ in range(rng.integers(4, 9)):
            _shape(img, rng)
        img = cv2.addWeighted(img, 0.85, np.full_like(img, rng.integers(0, 255, 3).tolist()), 0.15, 0)
        return img
    c1, c2 = rng.integers(0, 255, 3), rng.integers(0, 255, 3)
    t = np.linspace(0, 1, h)[:, None, None]
    img = ((1 - t) * c1 + t * c2).astype(np.float32) * np.ones((1, w, 1))
    img = img.astype(np.uint8)
    for _ in range(rng.integers(12, 30)):
        _shape(img, rng)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _shape(img, rng):
    h, w = img.shape[:2]
    col = rng.integers(0, 255, 3).tolist()
    k = rng.integers(0, 4)
    if k == 0:
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(10, h // 3)), col, -1)
    elif k == 1:
        p = rng.integers(0, [w, h], (2, 2))
        cv2.rectangle(img, tuple(map(int, p[0])), tuple(map(int, p[1])), col, -1)
    elif k == 2:
        pts = rng.integers(0, [w, h], (int(rng.integers(3, 7)), 2)).astype(np.int32)
        cv2.fillPoly(img, [pts], col)
    else:
        pts = rng.integers(0, [w, h], (int(rng.integers(3, 8)), 2)).astype(np.int32)
        cv2.polylines(img, [pts], False, col, int(rng.integers(3, 18)))


def _frame(img, rng):
    h, w = img.shape[:2]
    b = int(rng.uniform(0.04, 0.10) * max(h, w))
    gold = np.array([40, 140, 190], np.float32) * rng.uniform(0.6, 1.2) if rng.random() < 0.7 \
        else rng.integers(10, 80, 3).astype(np.float32)
    out = np.empty((h + 2 * b, w + 2 * b, 3), np.float32)
    yy = np.abs(np.linspace(-1, 1, h + 2 * b))[:, None]
    out[:] = gold * (0.75 + 0.35 * (1 - yy))[..., None]
    out[b:b + h, b:b + w] = img
    return np.clip(out, 0, 255).astype(np.uint8)


def museum_photo(ref: np.ndarray, rng: np.random.Generator, out_size: int = 800,
                 strength: float = 1.0) -> np.ndarray:
    s = strength
    img = _frame(ref, rng) if rng.random() < 0.85 else ref
    h, w = img.shape[:2]
    # mur
    W = out_size
    H = int(out_size * rng.uniform(0.7, 1.0))
    wall = np.full((H, W, 3), rng.integers(120, 235, 3), np.uint8)
    wall = np.clip(wall + rng.normal(0, 4, wall.shape), 0, 255).astype(np.uint8)
    # placement + perspective
    fill = rng.uniform(0.45, 0.9)
    sc = fill * min(W / w, H / h)
    tw, th = w * sc, h * sc
    cx, cy = W / 2 + rng.uniform(-0.12, 0.12) * W, H / 2 + rng.uniform(-0.12, 0.12) * H
    base = np.array([[-tw / 2, -th / 2], [tw / 2, -th / 2], [tw / 2, th / 2], [-tw / 2, th / 2]])
    yaw = rng.uniform(-0.35, 0.35) * s  # rotation autour de l'axe vertical : bords gauche/droit inégaux
    pitch = rng.uniform(-0.15, 0.15) * s
    dst = base.copy()
    dst[[0, 3], 1] *= 1 - yaw
    dst[[1, 2], 1] *= 1 + yaw
    dst[[0, 1], 0] *= 1 - pitch
    dst[[2, 3], 0] *= 1 + pitch
    dst += rng.normal(0, 0.015 * s * max(tw, th), dst.shape)
    rot = rng.uniform(-4, 4) * s * np.pi / 180
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    dst = dst @ R.T + [cx, cy]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    photo = np.where(mask[..., None] > 0, warped, wall).astype(np.float32)
    # éclairage : dégradé + teinte (lumière artificielle)
    gx = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.8, 1.15), W)[None, :, None]
    gy = np.linspace(rng.uniform(0.75, 1.1), rng.uniform(0.6, 1.0), H)[:, None, None]
    tint = np.array([rng.uniform(0.85, 1.0), 1.0, rng.uniform(1.0, 1.15)])
    photo *= gx * gy * tint * rng.uniform(0.55, 1.05) ** s
    # reflet (verre)
    if rng.random() < 0.6 * s:
        glare = np.zeros((H, W), np.float32)
        cv2.ellipse(glare, (int(rng.uniform(0.2, 0.8) * W), int(rng.uniform(0.2, 0.8) * H)),
                    (int(rng.uniform(0.05, 0.2) * W), int(rng.uniform(0.03, 0.1) * H)),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), rng.uniform(8, 30))
        photo += glare[..., None] * rng.uniform(80, 200)
    # occlusion (visiteur)
    if rng.random() < 0.35 * s:
        ox = int(rng.uniform(0.0, 0.85) * W)
        ow = int(rng.uniform(0.1, 0.3) * W)
        top = int(rng.uniform(0.45, 0.8) * H)
        col = rng.integers(10, 90, 3).tolist()
        cv2.ellipse(photo, (ox + ow // 2, top), (ow // 3, ow // 3), 0, 0, 360, col, -1)
        cv2.rectangle(photo, (ox, top + ow // 4), (ox + ow, H), col, -1)
    # flou, bruit capteur, JPEG
    if rng.random() < 0.7:
        k = rng.uniform(0.5, 2.2) * s
        photo = cv2.GaussianBlur(photo, (0, 0), k)
    photo += rng.normal(0, rng.uniform(2, 9) * s, photo.shape)
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    q = int(rng.uniform(45, 90))
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
