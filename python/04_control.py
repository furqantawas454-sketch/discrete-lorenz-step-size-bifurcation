"""04 -- Section 6 and Example 7.5: state feedback control, Jury conditions, controlled simulations."""
import numpy as np
import lorenz_map as L
from report import Report

R = Report("04_control")
s, b = L.SIGMA, L.BETA
rng = np.random.default_rng(3)

R.line("== Theorem 6.1: controlled characteristic polynomial ==")
ok = True
for _ in range(300):
    k1, k2 = rng.uniform(-2, 2, 2); h = rng.uniform(0.001, 0.3); rho = rng.uniform(1.01, 40)
    cp = np.poly(L.J_controlled(k1, k2, h, rho))
    a2, a1, a0 = L.control_coeffs(k1, k2, h, rho)
    p1 = b * h ** 2 * (rho * k1 + (2 - rho) * k2 + 2 * s * (rho - 1) * h)
    mp1 = (8 - 4 * k1 - 4 * h * (s + b + 1) + 2 * h * (b + 1) * k1 + 2 * h * k2
           + b * h ** 2 * (2 * (s + rho) - rho * k1 + (rho - 2) * k2) - 2 * s * b * (rho - 1) * h ** 3)
    ok &= np.allclose(cp, [1, a2, a1, a0]) and np.isclose(1 + a2 + a1 + a0, p1) and np.isclose(-(-1 + a2 - a1 + a0), mp1)
R.check("coefficients (6.2), p(1) of (6.3) and -p(-1) of (6.4) agree with det(mu I - J_c)  (300 random cases)", ok)
ok = True
for _ in range(100):
    k1, k2 = rng.uniform(-2, 2, 2); h = rng.uniform(0.001, 0.3); rho = rng.uniform(1.01, 40)
    Jp = L.J_controlled(k1, k2, h, rho)
    Jm = L.J(L.Em(rho), h, rho) - np.outer([1, 0, 0], [k1, k2, 0])
    ok &= np.allclose(np.sort_complex(np.linalg.eigvals(Jp)), np.sort_complex(np.linalg.eigvals(Jm)))
R.check("the same gains give the same spectrum at E_-  (100 random cases)", ok)

R.line("\n== Example 7.5 ==")
for (rho, h, k1, k2, r0, rc, n_on) in [(10, 0.07, 0.5, 0.5, 1.0513, 0.9575, 3009), (28, 0.01, 0.5, 0.0, 1.0061, 0.9877, 3717)]:
    K1, K2 = np.meshgrid(np.linspace(-1, 2, 301), np.linspace(-2, 2, 401))
    jy = L.jury(K1, K2, h, rho)
    rad = np.array([[max(abs(np.linalg.eigvals(L.J_controlled(a, c, h, rho)))) for a, c in zip(r1, r2)] for r1, r2 in zip(K1, K2)])
    mism = (rad < 1) != jy
    R.line(f"  (rho, h) = ({rho}, {h}): grid 301 x 401, mismatches {mism.sum()}, largest |rad - 1| at mismatches {np.abs(rad[mism] - 1).max() if mism.any() else 0:.1e}")
    R.check("Jury region = eigenvalue region, mismatches only on the boundary", (not mism.any()) or np.abs(rad[mism] - 1).max() < 1e-12)
    rad0 = max(abs(np.linalg.eigvals(L.J(L.Ep(rho), h, rho)))); radc = max(abs(np.linalg.eigvals(L.J_controlled(k1, k2, h, rho))))
    R.line(f"  spectral radius without control {rad0:.4f}, with gains ({k1}, {k2}) {radc:.4f}")
    R.check(f"spectral radii {r0} -> {rc}, gains satisfy the Jury conditions", abs(rad0 - r0) < 1e-4 and abs(radc - rc) < 1e-4 and bool(L.jury(k1, k2, h, rho)))
    E = L.Ep(rho); X = np.array([1.0, 1.0, 1.0]); on = None
    for n in range(12000):
        u = 0.0
        if on is None and n >= 3000 and np.linalg.norm(X - E) < 3: on = n
        if on is not None: u = -k1 * (X[0] - E[0]) - k2 * (X[1] - E[1])
        X = L.G(X, h, rho); X[0] += u
    R.line(f"  control switched on at n = {on}; final distance from E_+ {np.linalg.norm(X - E):.1e}")
    R.check(f"switch-on at n = {n_on} and convergence to E_+ (distance < 1e-13)", on == n_on and np.linalg.norm(X - E) < 1e-13)
R.finish()
