"""03 -- Section 7.2: every number quoted in Examples 7.1-7.4 (orbits, scaling laws, MLE, bistability)."""
import warnings
import numpy as np
from scipy.optimize import fsolve
warnings.filterwarnings('ignore')
import lorenz_map as L
from report import Report

np.seterr(all="ignore")
R = Report("03_numerical_examples")
G, Ep, S = L.G, L.Ep, L.S

R.line("== Example 7.1 (rho = 0.5): subcritical flip at the origin ==")
rho = 0.5; h1 = L.h_origin(rho)[0]
amps, mults = [], []
X = np.array([0.7265, -0.0389, -0.0106])
for eps in [1e-3, 2.5e-4]:
    h = h1 * (1 - eps)
    X = fsolve(lambda U: G(U, h, rho) - S @ U, X if eps == 1e-3 else X / 2, xtol=1e-13)
    M = L.J(G(X, h, rho), h, rho) @ L.J(X, h, rho)
    amps.append(abs(X[0])); mults.append(max(abs(np.linalg.eigvals(M))))
    R.line(f"  h = h1(1 - {eps:g}): 2-cycle through x = +-{abs(X[0]):.4f}, G(X) = SX error {np.linalg.norm(G(X, h, rho) - S @ X):.1e}, largest |multiplier| {mults[-1]:.4f}")
R.check("amplitudes 0.7265 and 0.3657, ratio ~ 2 (square-root law)", abs(amps[0] - 0.7265) < 1e-4 and abs(amps[1] - 0.3657) < 1e-4 and abs(amps[0] / amps[1] - 2) < 0.02)
R.check("the 2-cycle is unstable (|multiplier| = 1.008 > 1)", mults[0] > 1 and abs(mults[0] - 1.008) < 1e-3)
Xc = np.array([0.7265, -0.0389, -0.0106])
for h in np.linspace(h1 * (1 - 1e-3), 0.17, 300):
    Xc = fsolve(lambda U: G(U, h, rho) - S @ U, Xc, xtol=1e-13)
R.line(f"  continued to h = 0.17: |x| = {abs(Xc[0]):.2f}")
R.check("|x| = 4.60 at h = 0.17", abs(abs(Xc[0]) - 4.60) < 0.01)
R.check("orbit from (0.01,0,0) converges for h < h1 and escapes for h > h1",
        np.linalg.norm(L.orbit(rho, 0.999 * h1, [0.01, 0, 0], 20000, 1)[0]) < 1e-6 and L.orbit(rho, 1.001 * h1, [0.01, 0, 0], 20000, 1) is None)

R.line("\n== Example 7.2 (rho = 2): supercritical flip at E_+ ==")
rho = 2.0; hF = L.h_F(rho); E = Ep(rho)
sep = []
for eps in [1e-3, 2.5e-4]:
    o = L.orbit(rho, hF * (1 + eps), E + [1e-3, 1e-3, 1e-3], 200000, 3)
    sep.append(np.linalg.norm(o[0] - o[1])); R.line(f"  h = h_F(1 + {eps:g}): period-2 error {np.linalg.norm(o[2] - o[0]):.1e}, distance between the two points {sep[-1]:.4f}")
R.check("stable 2-cycle with point separation 0.9582 and 0.4799 (square-root law)", abs(sep[0] - 0.9582) < 1e-3 and abs(sep[1] - 0.4799) < 1e-3)
o = L.orbit(rho, 0.183, E + [0.01, 0, 0], 40000, 2)
R.line(f"  h = 0.183: 2-cycle points (x,y) = ({o[0,0]:.3f},{o[0,1]:.3f}), ({o[1,0]:.3f},{o[1,1]:.3f})")
R.check("2-cycle points (-0.156, 2.405) and (4.531, 1.970)", np.allclose(sorted(o[:, 0]), [-0.156, 4.531], atol=1e-3))
hs = np.linspace(0.170, 0.1855, 1400)
_, _, lam, alive = L.bifurcation_scan(rho, hs, E + [0.01, 0, 0], transient=40000, mle_n=30000)
hesc = hs[np.where(~alive)[0][0]]
R.line(f"  first escape on the grid: h = {hesc:.4f};  MLE at h=0.1843: {L.mle(rho, 0.1843, E + [0.01, 0, 0], 40000, 40000):.5f}")
R.check("escape at h ~ 0.1844, secondary invariant curves (MLE ~ 0) at h = 0.1843",
        abs(hesc - 0.1844) < 2e-4 and abs(L.mle(rho, 0.1843, E + [0.01, 0, 0], 40000, 40000)) < 1e-3)

R.line("\n== Example 7.3 (rho = 5): supercritical Neimark--Sacker ==")
rho = 5.0; hN = L.h_NS(rho); E = Ep(rho)
rng_ = []
for eps in [5e-4, 2e-3]:
    o = L.orbit(rho, hN * (1 + eps), E + [1e-3, -5e-4, 3e-4], 400000, 4000)
    dd = np.linalg.norm(o - E, axis=1); rng_.append((dd.min(), dd.max()))
    R.line(f"  h = h_NS(1 + {eps:g}): distance from E_+ in [{dd.min():.3f}, {dd.max():.3f}]")
R.check("invariant curve sizes [0.189,0.285] and [0.372,0.572]", np.allclose(rng_[0], (0.189, 0.285), atol=2e-3) and np.allclose(rng_[1], (0.372, 0.572), atol=2e-3))
hs = np.linspace(0.095, 0.1575, 1400)
xs, _, lam, alive = L.bifurcation_scan(rho, hs, E + [0.01, 0, 0], transient=30000, mle_n=30000)
hmerge = hs[np.where(np.nanmin(xs, axis=0) < -1)[0][0]]; hesc = hs[np.where(~alive)[0][0]]
R.line(f"  merger of the two curves at h = {hmerge:.4f}; escape at h = {hesc:.4f}")
R.check("merger at h ~ 0.1238 and escape at h ~ 0.1573", abs(hmerge - 0.1238) < 2e-4 and abs(hesc - 0.1573) < 2e-4)
hs2 = np.linspace(0.1234, 0.1242, 81)
_, _, lam2, _ = L.bifurcation_scan(rho, hs2, [3.276, 3.276, 4.01], transient=30000, keep=10, mle_n=60000)
R.line(f"  largest MLE near the merger: {np.nanmax(lam2):.4f} at h = {hs2[np.nanargmax(lam2)]:.5f}")
R.check("thin chaotic layer at the merger: MLE up to ~0.046 near h = 0.1237", abs(np.nanmax(lam2) - 0.046) < 0.002)
A = L.orbit(rho, 0.154, E + [0.01, 0, 0], 80000, 26); Bo = L.orbit(rho, 0.154, S @ E + [-0.01, 0, 0], 80000, 26)
per = next(p for p in range(1, 26) if np.linalg.norm(A[p] - A[0]) < 1e-7)
R.check("h = 0.154: period-25 orbit, and the orbit from E_- is its mirror image",
        per == 25 and min(np.linalg.norm((S @ A.T).T - Bo[0], axis=1)) < 1e-8 and min(np.linalg.norm(A - Bo[0], axis=1)) > 0.1)

R.line("\n== Example 7.4 (rho = 10): subcritical Neimark--Sacker, discretization-induced chaos ==")
rho = 10.0; hN = L.h_NS(rho); E = Ep(rho)
o = L.orbit(rho, hN * (1 + 2e-3), E + [1e-3, 0, 0], 400000, 20000)
dd = np.linalg.norm(o - E, axis=1)
R.line(f"  h = h_NS(1 + 0.002): distance from E_+ in [{dd.min():.2f}, {dd.max():.2f}]")
R.check("jump to a distant attractor (distance 4.5 to 20.9)", dd.min() > 4 and dd.max() > 20)
hs = np.linspace(0.020, 0.0875, 1400)
_, _, lamA, _ = L.bifurcation_scan(rho, hs, E + [0.01, 0, 0], transient=40000, keep=10, mle_n=30000)
_, _, lamB, aliveB = L.bifurcation_scan(rho, hs, [1.0, 1.0, 1.0], transient=40000, keep=10, mle_n=30000)
onA = hs[np.where(np.nan_to_num(lamA, nan=-1) > 0.002)[0][0]]; onB = hs[np.where(np.nan_to_num(lamB, nan=-1) > 0.002)[0][0]]
hesc = hs[np.where(~aliveB)[0][0]]
R.line(f"  chaos onset: from near E_+ h = {onA:.4f}, from (1,1,1) h = {onB:.4f}; escape h = {hesc:.4f}")
R.check("onsets 0.0311 (near E_+) and 0.0279 (from (1,1,1)), escape 0.0816",
        abs(onA - 0.0311) < 2e-4 and abs(onB - 0.0279) < 2e-4 and abs(hesc - 0.0816) < 2e-4)
m1 = L.mle(rho, 0.03, [1, 1, 1], 40000, 40000); fin = L.orbit(rho, 0.03, E + [0.01, 0, 0], 40000, 1)[0]
R.check("bistability at h = 0.03: chaotic attractor (MLE > 0) coexists with stable E_+",
        m1 > 0.005 and np.linalg.norm(fin - E) < 1e-6)
m07 = [L.mle(rho, 0.07, X0, 40000, 1000000) for X0 in ([1, 1, 1], [2, -1, 5], [-3, 0.5, 12])]
R.line(f"  MLE at h = 0.07 over 1e6 iterates from three initial points: {', '.join(f'{v:.4f}' for v in m07)}  (mean {np.mean(m07) / 0.07:.2f} per unit time)")
R.check("MLE(0.07) ~ 0.043 per iteration, ~0.61 per unit time", all(abs(v - 0.043) < 5e-4 for v in m07))
R.finish()
