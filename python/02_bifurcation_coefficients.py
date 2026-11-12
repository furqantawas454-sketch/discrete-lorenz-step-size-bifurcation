"""02 -- Section 5 (and Section 7.1): transversality, normal form coefficients, resonances, rho_G."""
import numpy as np
from scipy.optimize import brentq
import lorenz_map as L
from report import Report

R = Report("02_bifurcation_coefficients")
s, b = L.SIGMA, L.BETA

R.line("== Lemma 5.1: transversality ==")
ok = True
for rho in [5, 10, 20]:
    lam = L.flow_eigs_Ep(rho); lam = lam[np.argmax(lam.imag)]
    hc = L.h_of_lambda(lam); e = 1e-7
    deriv = (abs(1 + (hc + e) * lam) - abs(1 + (hc - e) * lam)) / (2 * e)
    ok &= np.isclose(deriv, -lam.real, rtol=1e-6) and np.isclose(-lam.real, hc * abs(lam) ** 2 / 2)
R.check("d|mu|/dh = -a = h_c |lambda|^2 / 2 > 0 at the Neimark--Sacker threshold (rho = 5, 10, 20)", ok)

R.line("\n== Theorem 5.1: flip at the origin ==")
rho = 0.5
lm, _ = L.lambda_pm_origin(rho); h1, h3, _ = L.h_origin(rho)
q = np.array([1.0, (s + lm) / s, 0.0])
l0c = L.ell0_closed(rho)
l0g = L.flip_coefficient(np.zeros(3), h1, rho, unit_q=False, q=q)
l0u = L.flip_coefficient(np.zeros(3), h1, rho, unit_q=True, q=q)
R.line(f"  rho = 0.5: ell_0 (closed form, eq. (5.4)) = {l0c:.6f}; general formula, same q: {l0g:.6f}; unit q: {l0u:.6f}")
R.check("closed form (5.4) agrees with the general flip formula", np.isclose(l0c, l0g))
R.check("unit-q value equals closed form divided by |q|^2", np.isclose(l0u, l0c / (q @ q)))
vals = [L.ell0_closed(r) for r in np.linspace(0.01, 0.99, 99)] + [L.ell0_closed(0.5, 1, 3), L.ell0_closed(0.3, 3, 1)]
R.check("ell_0 < 0 on a grid of rho in (0,1) and for other (sigma, beta)", max(vals) < 0)
# degenerate case: (sigma, beta, rho) = (1, 3, 0.5), h3 < h1
h1d, h3d, _ = L.h_origin(0.5, 1, 3)
pts = [L.G(L.G([0, 0, z], h3d, 0.5, 1, 3), h3d, 0.5, 1, 3) for z in [0.1, -0.4, 2.0]]
R.check("Theorem 5.1(ii): for (1,3,0.5), h3 < h1 and every z-axis point is 2-periodic at h = h3",
        h3d < h1d and all(np.allclose(p, [0, 0, z]) for p, z in zip(pts, [0.1, -0.4, 2.0])))

R.line("\n== Theorem 5.2: flip at E_+- ==")
for rho, expect in [(1.2, 0.010314), (2.0, 0.008676), (3.0, 0.007202)]:
    R.line(f"  rho = {rho}: h_F = {L.h_F(rho):.5f}, ell_F = {L.ell_F(rho):.6f}")
R.check("ell_F(2) = 0.00868 (Example 7.2)", abs(L.ell_F(2.0) - 0.008676) < 1e-6)
rsw = brentq(lambda r: L.h_F(r) - L.h_NS(r), 2.5, 5)
lF = np.array([L.ell_F(r) for r in np.linspace(1.01, rsw, 200)])
R.line(f"  ell_F on (1, rho_sw): min {lF.min():.4f}, max {lF.max():.4f}")
R.check("ell_F > 0 on the whole flip-first branch, range [0.0065, 0.0108]", lF.min() > 0 and abs(lF.min() - 0.0065) < 1e-4 and abs(lF.max() - 0.0108) < 1e-4)

R.line("\n== Theorem 5.3: Neimark--Sacker at E_+- ==")
for rho in [5.0, 10.0]:
    d, h, th = L.d_NS(rho); m = L.m_of_h(h, rho)
    R.line(f"  rho = {rho}: h_NS = {h:.5f}, theta_0 = {th:.4f}, h^2 m = {h * h * m:.4f}, d = {d:.4e}")
R.check("d(5) = -1.63e-3 (supercritical) and d(10) = +3.79e-5 (subcritical)",
        abs(L.d_NS(5.0)[0] + 1.6278e-3) < 1e-7 and abs(L.d_NS(10.0)[0] - 3.7923e-5) < 1e-9)
rhoG = brentq(lambda r: L.d_NS(r)[0], 7, 10)
R.line(f"  generalized Neimark--Sacker point: rho_G = {rhoG:.4f}, h_NS = {L.h_NS(rhoG):.6f}")
R.check("rho_G = 8.6219, h = 0.040957", abs(rhoG - 8.6219) < 1e-4 and abs(L.h_NS(rhoG) - 0.040957) < 1e-6)
rr = np.linspace(rsw + 1e-3, L.rho_H() - 1e-3, 800)
dd = np.array([L.d_NS(r)[0] for r in rr]); hm = np.array([L.h_NS(r) ** 2 * L.m_of_h(L.h_NS(r), r) for r in rr])
R.check("d < 0 on (rho_sw, rho_G) and d > 0 on (rho_G, rho_H)", np.all(dd[rr < rhoG - 1e-3] < 0) and np.all(dd[rr > rhoG + 1e-3] > 0))
R.line(f"  max of h_NS^2 m on the Neimark--Sacker-first branch: {hm.max():.4f}")
R.check("non-resonance (5.6): h_NS^2 m <= 0.362 < 2 everywhere, no strong resonance", hm.max() < 0.3621)
R.finish()
