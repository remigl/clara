"""Banc d'essai : recherche brute (exacte) int8 vs float32, à l'échelle d'un musée et de l'Europe.

Usage : python3 tools/bench_search.py
Ne nécessite que numpy. Les temps mesurés ici sont ceux d'un CPU de bureau/serveur ;
sur un téléphone milieu de gamme (Cortex-A78/A55, NEON) compter environ 3 à 6x plus lent.

Note : en numpy, int8 est plus lent que f32 (conversion int16, pas de BLAS int8).
En natif (instructions ARM SDOT / dotprod), int8 est au contraire 2 à 4x plus rapide que f32.
"""
import time
import numpy as np

DIM = 512          # dimension typique d'un embedding CLIP/DINO réduit
VIEWS = 3          # vues de référence par œuvre
rng = np.random.default_rng(0)


def make(n):
    x = rng.standard_normal((n, DIM), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def quant_int8(x):
    return np.clip(np.round(x * 127), -127, 127).astype(np.int8)


def bench(n_works):
    n = n_works * VIEWS
    db = make(n)
    q = db[12345 % n] + 0.05 * make(1)[0]
    q /= np.linalg.norm(q)

    db8 = quant_int8(db)
    q8 = quant_int8(q[None])[0]

    def best(f, reps=20):
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
        return min(ts) * 1000, r

    t32, s32 = best(lambda: db @ q)
    t8, s8 = best(lambda: db8.astype(np.int16) @ q8.astype(np.int16))
    target = 12345 % n
    hit32 = int(np.argmax(s32)) == target
    hit8 = int(np.argmax(s8)) == target
    print(f"{n_works:>8} œuvres ({n:>8} vecteurs) | "
          f"f32 {db.nbytes/1e6:7.1f} Mo {t32:7.2f} ms | "
          f"int8 {db8.nbytes/1e6:7.1f} Mo {t8:7.2f} ms | top-1 correct f32={hit32} int8={hit8}")


if __name__ == "__main__":
    for n in (1_500, 10_000, 50_000, 300_000):
        bench(n)
