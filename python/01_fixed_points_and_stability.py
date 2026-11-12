"""01 -- Sections 2-4: fixed points, symmetry, characteristic polynomials and stability thresholds."""
import numpy as np
import lorenz_map as L
from report import Report

rng = np.random.default_rng(1)
R = Report("01_fixed_points_and_stability")
s, b = L.SIGMA, L.BETA

R.line("== Section 2: basic properties of the Euler--Lorenz map ==")
ok = True
for _ in range(200):
    X = rng.normal(size=3) * 5; h = rng.uniform(0.001, 0.5); rho = rng.uniform(0.1, 30)
    ok &= np.allclose(L.G(L.S @ X, h, rho), L.S @ L.G(X, h, rho))
R.check("Lemma 2.1: G_h(SX) = S G_h(X) at 200 random points", ok)
ok = all(np.allclose(L.G([0, 0, z], h, 5.0), [0, 0, (1 - h * b) * z]) for z in [-3, 0.7, 9] for h in [0.01, 0.3, 1.1])
R.check("Lemma 2.2: the z-axis is invariant and z -> (1 - h beta) z on it", ok)

R.line("\n== Section 3: fixed points and characteristic polynomials ==")
ok = True
for rho in [0.5, 1.5, 10, 28]:
    for h in [0.01, 0.2]:
        ok &= np.allclose(L.G([0, 0, 0], h, rho), 0)
        if rho > 1:
            ok &= np.allclose(L.G(L.Ep(rho), h, rho), L.Ep(rho)) and np.allclose(L.G(L.Em(rho), h, rho), L.Em(rho))
R.check("Theorem 3.1: E_0 and E_+- (rho > 1) are fixed points for every h", ok)
ok = True
for rho in [1.2, 5, 10, 28]:
    ev = np.sort_complex(np.linalg.eigvals(L.DF(L.Ep(rho), rho)))
    ok &= np.allclose(ev, np.sort_complex(L.flow_eigs_Ep(rho)), atol=1e-9)
R.check("Lemma 3.1(ii): eigenvalues of DF(E_+) are the roots of P, eq. (3.5)", ok)
ok = True
for rho in [0.5, 5, 28]:
    for h in [0.003, 0.07, 0.4]:
        X = L.Ep(rho) if rho > 1 else np.zeros(3)
        ok &= np.allclose(np.sort_complex(np.linalg.eigvals(L.J(X, h, rho))),
                          np.sort_complex(1 + h * np.linalg.eigvals(L.DF(X, rho))))
R.check("Eq. (3.3): multipliers are mu = 1 + h lambda", ok)

R.line("\n== Section 4: stability thresholds ==")
ok = True
for _ in range(2000):
    lam = complex(rng.normal() * 5, rng.normal() * 5); h = rng.uniform(0.001, 2)
    inside = abs(1 + h * lam) < 1
    pred = lam.real < 0 and h < L.h_of_lambda(lam)
    ok &= (inside == pred)
R.check("Lemma 4.1: |1 + h lambda| < 1  <=>  Re lambda < 0 and h < h(lambda)  (2000 random cases)", ok)

def spec_rad(X, h, rho, s_=s, b_=b):
    return max(abs(np.linalg.eigvals(L.J(X, h, rho, s_, b_))))

for (s_, b_, rho, expect) in [(10, 8 / 3, 0.5, 0.19002), (1, 3, 0.5, 2 / 3)]:
    h1, h3, h0 = L.h_origin(rho, s_, b_)
    R.line(f"  origin, (sigma,beta,rho)=({s_},{b_:.4g},{rho}): h1={h1:.5f}, h3={h3:.5f}, h0={h0:.5f}")
    R.check(f"Theorem 4.1: h0 = {expect:.5f} and spectral radius crosses 1 at h0",
            abs(h0 - expect) < 1e-5 and spec_rad(np.zeros(3), 0.999 * h0, rho, s_, b_) < 1 < spec_rad(np.zeros(3), 1.001 * h0, rho, s_, b_))
ok = True
for _ in range(200):
    h = rng.uniform(0.01, 1); rho = rng.uniform(1.01, 30)
    ok &= np.isclose(h ** 3 * np.polyval(L.P_coeffs(rho), -2 / h), 2 * L.F_poly(h, rho))
R.check("Lemma 4.2(i): h^3 P(-2/h) = 2 F(h)  (200 random cases)", ok)
ok = True
for rho in np.linspace(1.01, 30, 60):
    c, Bc, K = s * b * (rho - 1), b * (s + rho), s + b + 1
    ok &= np.isclose(L.N_poly(0.0, rho), 2 * c - K * Bc) and np.isclose(L.N_poly(0.0, rho), b * (rho * (s - b - 1) - s * (s + b + 3)))
R.check("Remark 2: N(0) = 2c - KB = beta[rho(sigma-beta-1) - sigma(sigma+beta+3)]", ok)
R.check("Remark 2: N(0) vanishes exactly at rho_H = 470/19", abs(L.N_poly(0.0, L.rho_H())) < 1e-10 and abs(L.rho_H() - 470 / 19) < 1e-12)

R.line("\n  rho      h_F (from F)   min over real   h_NS (from N)   min over complex   spec. rad. at 0.999/1.001 h_E")
cases = [(10, 8 / 3, r) for r in [1.2, 2, 5, 10, 15, 20, 24]] + [(16, 4, 30), (3, 1, 2)]
ok_all = True
for (s_, b_, rho) in cases:
    lam = np.roots(L.P_coeffs(rho, s_, b_))
    real = lam[np.abs(lam.imag) < 1e-12].real; cplx = lam[np.abs(lam.imag) > 1e-12]
    hF, hN = L.h_F(rho, s_, b_), L.h_NS(rho, s_, b_)
    hF2 = min(2 / np.abs(real)); hN2 = min(L.h_of_lambda(cplx)) if len(cplx) else np.inf
    hE = min(hF, hN)
    r1, r2 = spec_rad(L.Ep(rho, b_), 0.999 * hE, rho, s_, b_), spec_rad(L.Ep(rho, b_), 1.001 * hE, rho, s_, b_)
    ok = np.isclose(hF, hF2) and (np.isclose(hN, hN2) or (hN == np.inf and hN2 == np.inf)) and r1 < 1 < r2
    ok_all &= ok
    R.line(f"  {rho:5.1f}   {hF:.5f}        {hF2:.5f}         {hN:.5f}         {hN2:.5f}            {r1:.9f} / {r2:.9f}" + ("" if (s_, b_) == (10, 8 / 3) else f"   (sigma,beta)=({s_},{b_})"))
R.check("Theorem 4.2: h_E = min(h_F, h_NS) from F and N agrees with the eigenvalues, 9 parameter sets", ok_all)

from scipy.optimize import brentq
def disc(rho):
    a, bb, c, d = L.P_coeffs(rho)
    return 18 * a * bb * c * d - 4 * bb ** 3 * d + bb ** 2 * c ** 2 - 4 * a * c ** 3 - 27 * a ** 2 * d ** 2
rho1 = brentq(disc, 1.01, 2.0)
rsw = brentq(lambda r: L.h_F(r) - L.h_NS(r), 2.5, 5)
R.line(f"\n  rho_1 (complex pair appears) = {rho1:.4f},  rho_sw = {rsw:.4f} with h_F = h_NS = {L.h_F(rsw):.5f}")
R.check("Section 7.1: rho_1 = 1.3456 and rho_sw = 3.6288, h = 0.17281", abs(rho1 - 1.3456) < 1e-4 and abs(rsw - 3.6288) < 1e-4 and abs(L.h_F(rsw) - 0.17281) < 1e-5)
R.finish()
