# random_params.py

class Randomizer:

    @staticmethod
    def _randomize_params(parent: dict | None = None) -> dict:
        import math, random

        # ---- Family scale: discrete A0 multiples ----
        if parent and "A0_base" in parent:
            A0_base = parent["A0_base"]
        else:
            A0_base = random.choice([0.40, 0.50, 0.60])
        m_int = random.choice([2, 3])            # integer multiple
        A0 = m_int * A0_base                      # your “integer multiple of parent A0”

        # ---- Petal count & spectrum layout ----
        P = random.choice([4, 5, 6, 7, 8, 10, 12])    # petals / symmetry
        multiples = random.choice([[1, 2], [1, 2, 3]])  # how many anchored harmonics
        ks_core = [j * P for j in multiples]           # {P, 2P, 3P}
        wobble = []
        if random.random() < 0.7:
            wobble.append(max(2, P - 1))
        if random.random() < 0.6:
            wobble.append(P + 1)
        ks = sorted(set(ks_core + wobble))

        # ---- Amplitude budget & decay ----
        S = random.uniform(0.20, 0.40)        # total ripple vs A0
        target_sum = S * A0
        p = random.uniform(1.1, 2.0)          # decay exponent
        # base weights ~ (k/P)^(-p)
        weights = []
        for k in ks:
            rel = max(1.0, k / float(P))
            weights.append(rel ** (-p))
        wsum = sum(weights) or 1.0
        # distribute amplitudes
        As = [target_sum * (w / wsum) for w in weights]

        # ---- Spikiness boost on the fundamental ----
        if P in ks:
            i = ks.index(P)
            As[i] *= random.uniform(1.2, 1.8)
            As[i] = min(As[i], 0.28 * A0)  # safety clamp

        # ---- Phase strategy ----
        phis = []
        for k in ks:
            if k % P == 0:  # anchored harmonic → near 0 or pi
                base = 0.0 if random.random() < 0.5 else math.pi
                jitter = random.uniform(-0.15, 0.15)   # small jitter
                phis.append(base + jitter)
            else:           # wobble → fully random
                phis.append(random.uniform(0, 2 * math.pi))

        terms = [{"k": int(k), "A": float(A), "phi": float(phi)}
                 for k, A, phi in zip(ks, As, phis)]

        params = {"A0": float(A0), "P": int(P), "A0_base": float(A0_base), "terms": terms}

        # ---- Radius floor check (uniform downscale if needed) ----
        try:
            import cupy as cp
            theta = cp.linspace(0, 2 * cp.pi, 4096, endpoint=False, dtype=cp.float32)
            r = cp.full(theta.shape, params["A0"], dtype=cp.float32)
            for t in terms:
                r += t["A"] * cp.cos(t["k"] * theta + t["phi"])
            r_min = float(cp.min(r).get())
        except Exception:
            # Fallback approximate check on CPU if CuPy unavailable here
            import numpy as np
            theta = np.linspace(0, 2 * np.pi, 4096, endpoint=False, dtype=np.float32)
            r = np.full(theta.shape, params["A0"], dtype=np.float32)
            for t in terms:
                r += t["A"] * np.cos(t["k"] * theta + t["phi"])
            r_min = float(r.min())

        floor = 0.42 * A0
        if r_min < floor:
            scale = max(0.15, min(1.0, (A0 - floor) / max(A0 - r_min, 1e-6)))
            for t in terms:
                t["A"] *= scale

        return params
