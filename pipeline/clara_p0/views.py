"""Vues d'images : recadrages de référence et détection classique du tableau dans une photo.

En P0, la détection du tableau est une méthode classique (contours + quadrilatère convexe).
Le détecteur appris « 4 coins » de l'architecture (§4.2) la remplacera en P2.
"""
from __future__ import annotations

import cv2
import numpy as np

# Vues de référence : nom -> (x0, y0, x1, y1) en fractions de l'image
REFERENCE_VIEWS: dict[str, tuple[float, float, float, float]] = {
    "full": (0.0, 0.0, 1.0, 1.0),
    "crop_center": (0.1, 0.1, 0.9, 0.9),
    "crop_tl": (0.0, 0.0, 0.7, 0.7),
    "crop_tr": (0.3, 0.0, 1.0, 0.7),
    "crop_bl": (0.0, 0.3, 0.7, 1.0),
    "crop_br": (0.3, 0.3, 1.0, 1.0),
}
DEFAULT_REFERENCE_VIEWS = ("full", "crop_center")


def crop_frac(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[round(y0 * h): round(y1 * h), round(x0 * w): round(x1 * w)]


def reference_views(img: np.ndarray, names=DEFAULT_REFERENCE_VIEWS) -> list[tuple[str, np.ndarray]]:
    return [(n, crop_frac(img, REFERENCE_VIEWS[n])) for n in names]


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def detect_quad(img: np.ndarray, min_area_frac: float = 0.12) -> tuple[np.ndarray, float] | None:
    """Cherche le plus grand quadrilatère convexe. Retourne (coins TL,TR,BR,BL, score 0..1) ou None."""
    h, w = img.shape[:2]
    scale = 640 / max(h, w) if max(h, w) > 640 else 1.0
    small = cv2.resize(img, (round(w * scale), round(h * scale))) if scale != 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    area_img = gray.size
    best, best_area = None, 0.0
    # plusieurs seuils : les cadres dorés et les murs clairs n'ont pas le même contraste
    for blur in (5, 9):
        g = cv2.GaussianBlur(gray, (blur, blur), 0)
        for lo, hi in ((20, 60), (40, 120), (80, 200)):
            edges = cv2.dilate(cv2.Canny(g, lo, hi), np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                hull = cv2.convexHull(c)
                a = cv2.contourArea(hull)
                if a < min_area_frac * area_img or a > 0.98 * area_img or a <= best_area:
                    continue
                for eps in (0.02, 0.04, 0.06):
                    approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
                    if len(approx) == 4:
                        best, best_area = approx, a
                        break
    if best is None:
        return None
    corners = _order_corners(best) / scale
    return corners, min(1.0, best_area / area_img / 0.5)


def rectify(img: np.ndarray, corners: np.ndarray, inset: float = 0.04) -> np.ndarray:
    """Redresse le quadrilatère et rogne légèrement le bord (cadre)."""
    tl, tr, br, bl = corners
    wd = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    ht = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    wd, ht = max(wd, 16), max(ht, 16)
    dst = np.array([[0, 0], [wd - 1, 0], [wd - 1, ht - 1], [0, ht - 1]], np.float32)
    m = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    out = cv2.warpPerspective(img, m, (wd, ht))
    dx, dy = round(wd * inset), round(ht * inset)
    return out[dy: ht - dy, dx: wd - dx]


def query_views(img: np.ndarray) -> tuple[list[tuple[str, np.ndarray]], float]:
    """Vues d'une photo requête : rectifiée (si détectée), complète, centre. Retourne (vues, score_détection)."""
    views: list[tuple[str, np.ndarray]] = []
    det = detect_quad(img)
    det_score = 0.0
    if det is not None:
        corners, det_score = det
        views.append(("rectified", rectify(img, corners)))
    views.append(("full", img))
    views.append(("center", crop_frac(img, (0.15, 0.15, 0.85, 0.85))))
    return views, det_score
