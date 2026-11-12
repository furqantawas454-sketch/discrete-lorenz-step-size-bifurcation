"""lorenz_map.py -- shared definitions for the forward Euler discretization of the Lorenz system.

    x_{n+1} = x_n + h*sigma*(y_n - x_n)
    y_{n+1} = y_n + h*(x_n*(rho - z_n) - y_n)
    z_{n+1} = z_n + h*(x_n*y_n - beta*z_n)

Everything in the paper uses sigma = 10, beta = 8/3 unless stated otherwise.
Section/equation numbers refer to the manuscript.
"""
import numpy as np

SIGMA, BETA = 10.0, 8.0 / 3.0
S = np.diag([-1.0, -1.0, 1.0])          # the Lorenz symmetry (x, y, z) -> (-x, -y, z)


# ----------------------------------------------------------------- the map (Section 2)
def F(X, rho, s=SIGMA, b=BETA):
    x, y, z = X
    return np.array([s * (y - x), x * (rho - z) - y, x * y - b * z])


def G(X, h, rho, s=SIGMA, b=BETA):
    """One step of the Euler--Lorenz map G_h, eq. (2.4).
    Written out exactly as in the simulations of Section 7, so that chaotic orbits
    (which amplify round-off) are reproduced iterate by iterate."""
    x, y, z = X
    return np.array([x + h * s * (y - x), y + h * (x * (rho - z) - y), z + h * (x * y - b * z)])


def J(X, h, rho, s=SIGMA, b=BETA):
    """Jacobian of G_h, eq. (3.2)."""
    x, y, z = X
    return np.array([[1 - h * s, h * s, 0.0],
                     [h * (rho - z), 1 - h, -h * x],
                     [h * y, h * x, 1 - h * b]])


def DF(X, rho, s=SIGMA, b=BETA):
    x, y, z = X
    return np.array([[-s, s, 0.0], [rho - z, -1.0, -x], [y, x, -b]])


def Ep(rho, b=BETA):
    """The symmetric fixed point E_+, eq. (3.1)."""
    q = np.sqrt(b * (rho - 1))
    return np.array([q, q, rho - 1])


def Em(rho, b=BETA):
    return S @ Ep(rho, b)


def B(u, v, h):
    """Symmetric bilinear form of the exact Taylor expansion, eq. (5.2)."""
    return h * np.array([0.0, -(u[0] * v[2] + u[2] * v[0]), u[0] * v[1] + u[1] * v[0]])


# ------------------------------------------------------- thresholds (Sections 3-4)
def rho_H(s=SIGMA, b=BETA):
    return s * (s + b + 3) / (s - b - 1)


def lambda_pm_origin(rho, s=SIGMA):
    """lambda_- and lambda_+ of eq. (4.3)."""
    d = np.sqrt((s - 1) ** 2 + 4 * s * rho)
    return (-(s + 1) - d) / 2, (-(s + 1) + d) / 2


def h_origin(rho, s=SIGMA, b=BETA):
    """h_1, h_3 and h_0 = min(h_1, h_3) of eq. (4.4)."""
    lm, _ = lambda_pm_origin(rho, s)
    h1, h3 = 2 / abs(lm), 2 / b
    return h1, h3, min(h1, h3)


def P_coeffs(rho, s=SIGMA, b=BETA):
    """Coefficients of P(lambda) = lambda^3 + K lambda^2 + B lambda + 2c, eq. (3.5)."""
    return [1.0, s + b + 1, b * (s + rho), 2 * s * b * (rho - 1)]


def flow_eigs_Ep(rho, s=SIGMA, b=BETA):
    return np.roots(P_coeffs(rho, s, b))


def h_of_lambda(lam):
    """h(lambda) = -2 Re(lambda) / |lambda|^2 of Lemma 4.1."""
    return -2 * np.real(lam) / np.abs(lam) ** 2


def F_poly(h, rho, s=SIGMA, b=BETA):
    """F(h) of eq. (4.5)."""
    c, Bc, K = s * b * (rho - 1), b * (s + rho), s + b + 1
    return c * h ** 3 - Bc * h ** 2 + 2 * K * h - 4


def m_of_h(h, rho, s=SIGMA, b=BETA):
    return b * (s + rho) - 2 * s * b * (rho - 1) * h


def N_poly(h, rho, s=SIGMA, b=BETA):
    """N(h) of eq. (4.6)."""
    m = m_of_h(h, rho, s, b)
    return m * (h * m - (s + b + 1)) + 2 * s * b * (rho - 1)


def h_F(rho, s=SIGMA, b=BETA):
    """Smallest positive root of F (Lemma 4.2(i))."""
    c, Bc, K = s * b * (rho - 1), b * (s + rho), s + b + 1
    r = np.roots([c, -Bc, 2 * K, -4])
    r = r[(np.abs(r.imag) < 1e-12) & (r.real > 0)].real
    return r.min()


def h_NS(rho, s=SIGMA, b=BETA):
    """The admissible root of N (Lemma 4.2(ii)); +inf if none."""
    c, Bc, K = s * b * (rho - 1), b * (s + rho), s + b + 1
    # N(h) = (B - 2ch)(h(B - 2ch) - K) + 2c expanded in powers of h
    coeffs = [4 * c ** 2, -4 * Bc * c, Bc ** 2 + 2 * c * K, -Bc * K + 2 * c]
    r = np.roots(coeffs)
    r = r[(np.abs(r.imag) < 1e-10) & (r.real > 0)].real
    r = [h for h in r if m_of_h(h, rho, s, b) > 0 and h * h * m_of_h(h, rho, s, b) < 4]
    return min(r) if r else np.inf


# ----------------------------------------------- normal form coefficients (Section 5)
def ell0_closed(rho, s=SIGMA, b=BETA):
    """ell_0 of eq. (5.4), eigenvector normalized as q = (1, (s+lam_-)/s, 0)."""
    lm, _ = lambda_pm_origin(rho, s)
    h1 = 2 / abs(lm)
    return -(h1 / b) * (s + lm) ** 2 / (s * rho + (s + lm) ** 2)


def flip_coefficient(X, h, rho, unit_q=True, q=None, s=SIGMA, b=BETA):
    """-1/2 <p, B(q, (A-I)^{-1} B(q,q))> at a fixed point with multiplier -1 (eqs. (5.4), (5.5))."""
    A = J(X, h, rho, s, b)
    if q is None:
        w, V = np.linalg.eig(A)
        q = np.real(V[:, np.argmin(abs(w + 1))])
    q = np.asarray(q, float)
    if unit_q:
        q = q / np.linalg.norm(q)
    w2, W = np.linalg.eig(A.T)
    p = np.real(W[:, np.argmin(abs(w2 + 1))])
    p = p / (p @ q)
    return -0.5 * p @ B(q, np.linalg.solve(A - np.eye(3), B(q, q, h)), h)


def ell_F(rho, s=SIGMA, b=BETA):
    h = h_F(rho, s, b)
    return flip_coefficient(Ep(rho, b), h, rho, s=s, b=b)


def d_NS(rho, s=SIGMA, b=BETA):
    """First Lyapunov coefficient d of eq. (5.7); returns (d, h_NS, theta_0)."""
    h = h_NS(rho, s, b)
    A = J(Ep(rho, b), h, rho, s, b)
    w, V = np.linalg.eig(A)
    i = np.argmax(w.imag)
    mu = w[i]
    q = V[:, i] / np.linalg.norm(V[:, i])
    w2, W = np.linalg.eig(A.T)
    p = W[:, np.argmin(abs(w2 - np.conj(mu)))]
    p = p / np.conj(np.conj(p) @ q)              # <p, q> = conj(p).q = 1
    th = np.angle(mu)
    I = np.eye(3)
    t = (2 * np.conj(p) @ B(q, np.linalg.solve(I - A, B(q, np.conj(q), h)), h)
         + np.conj(p) @ B(np.conj(q), np.linalg.solve(np.exp(2j * th) * I - A, B(q, q, h)), h))
    return 0.5 * np.real(np.exp(-1j * th) * t), h, th


# ------------------------------------------------------ state feedback (Section 6)
def control_coeffs(k1, k2, h, rho, s=SIGMA, b=BETA):
    """a2, a1, a0 of eq. (6.2)."""
    a2 = k1 - 3 + h * (s + b + 1)
    a1 = 3 - 2 * k1 + h * ((b + 1) * k1 + k2 - 2 * (s + b + 1)) + b * h ** 2 * (s + rho)
    a0 = (k1 - 1 + h * ((s + b + 1) - (b + 1) * k1 - k2)
          - b * h ** 2 * ((s + rho) - rho * k1 + (rho - 2) * k2) + 2 * s * b * (rho - 1) * h ** 3)
    return a2, a1, a0


def jury(k1, k2, h, rho, s=SIGMA, b=BETA):
    """The four Jury conditions (6.3)-(6.6); works element-wise on arrays."""
    a2, a1, a0 = control_coeffs(k1, k2, h, rho, s, b)
    return ((1 + a2 + a1 + a0 > 0) & (-(-1 + a2 - a1 + a0) > 0)
            & (np.abs(a0) < 1) & (np.abs(a0 ** 2 - 1) > np.abs(a0 * a2 - a1)))


def J_controlled(k1, k2, h, rho, s=SIGMA, b=BETA):
    return J(Ep(rho, b), h, rho, s, b) - np.outer([1, 0, 0], [k1, k2, 0])


# ------------------------------------------------------------ simulation helpers
def orbit(rho, h, X0, transient, n, bound=1e4):
    """Iterate G_h; returns an (n, 3) array or None if the orbit escapes."""
    X = np.array(X0, float)
    out = np.empty((n, 3))
    for k in range(transient + n):
        X = G(X, h, rho)
        if not np.all(np.isfinite(X)) or np.abs(X).max() > bound:
            return None
        if k >= transient:
            out[k - transient] = X
    return out


def mle(rho, h, X0, transient=20000, n=20000, bound=1e4):
    """Maximal Lyapunov exponent per iteration (nan if the orbit escapes)."""
    X = np.array(X0, float)
    v = np.array([1.0, 0.3, -0.2]); v /= np.linalg.norm(v)
    acc = 0.0
    for k in range(transient + n):
        A = J(X, h, rho)
        X = G(X, h, rho)
        if not np.all(np.isfinite(X)) or np.abs(X).max() > bound:
            return np.nan
        if k >= transient:
            v = A @ v; nv = np.linalg.norm(v); acc += np.log(nv); v /= nv
    return acc / n


def bifurcation_scan(rho, hs, X0, transient=20000, keep=300, mle_n=20000, bound=1e4,
                     s=SIGMA, b=BETA):
    """Vectorized over h: last `keep` x- and z-values and the MLE for each h (nan = escape)."""
    np.seterr(all="ignore")
    h = np.asarray(hs, float); n = len(h)
    x = np.full(n, X0[0], float); y = np.full(n, X0[1], float); z = np.full(n, X0[2], float)
    alive = np.ones(n, bool)

    def check():
        bad = ~(np.isfinite(x) & np.isfinite(y) & np.isfinite(z)) | (np.abs(x) > bound) | (np.abs(y) > bound) | (np.abs(z) > bound)
        alive[bad] = False
        x[~alive] = 0; y[~alive] = 0; z[~alive] = 0

    def step(x, y, z):
        return x + h * s * (y - x), y + h * (x * (rho - z) - y), z + h * (x * y - b * z)

    for k in range(transient):
        x, y, z = step(x, y, z)
        if k % 50 == 0: check()
    check()
    vx, vy, vz = np.ones(n), 0.3 * np.ones(n), -0.2 * np.ones(n)
    acc = np.zeros(n); xs = np.empty((keep, n)); zs = np.empty((keep, n))
    for k in range(mle_n):
        jx = (1 - h * s) * vx + h * s * vy
        jy = h * (rho - z) * vx + (1 - h) * vy - h * x * vz
        jz = h * y * vx + h * x * vy + (1 - h * b) * vz
        x, y, z = step(x, y, z)
        nv = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2); nv[nv == 0] = 1
        acc += np.log(nv); vx, vy, vz = jx / nv, jy / nv, jz / nv
        if k % 50 == 0: check()
        if k >= mle_n - keep:
            xs[k - (mle_n - keep)] = x; zs[k - (mle_n - keep)] = z
    check()
    lam = acc / mle_n
    lam[~alive] = np.nan; xs[:, ~alive] = np.nan; zs[:, ~alive] = np.nan
    return xs, zs, lam, alive
