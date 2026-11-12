# Step-size-induced bifurcations and chaos in the Euler-discretized Lorenz system: verification code

This repository contains the Wolfram Mathematica and Python code behind every theorem check,
numerical example and figure in the manuscript

> **F. Ahmad, M. K. Jamil**: *Step-size-induced bifurcations and chaos in the Euler-discretized
> Lorenz system: stability, flip and Neimark–Sacker bifurcations, and feedback control*
> (manuscript in preparation).

The code follows the order of the paper, so each section can be checked on its own.

## The model

The Lorenz system is discretized with the forward Euler scheme, with step size `h`:

```
x_{n+1} = x_n + h*sigma*(y_n - x_n)
y_{n+1} = y_n + h*(x_n*(rho - z_n) - y_n)
z_{n+1} = z_n + h*(x_n*y_n - beta*z_n)
```

Unless stated otherwise, `sigma = 10` and `beta = 8/3`. The step size `h` is the bifurcation parameter.

| Example (Section 7.2) | rho | What happens | Critical step size |
|---|---|---|---|
| 7.1 | 0.5  | subcritical flip of the origin | h_1 = 0.19002 |
| 7.2 | 2    | supercritical flip of E± | h_F = 0.17790 |
| 7.3 | 5    | supercritical Neimark–Sacker of E± | h_NS = 0.10282 |
| 7.4 | 10   | subcritical Neimark–Sacker, chaos created by the discretization | h_NS = 0.03096 |
| 7.5 | 10, 28 | state feedback control at (rho, h) = (10, 0.07) and (28, 0.01) | gains (0.5, 0.5), (0.5, 0) |

## Repository layout

```
mathematica/
  00_model.wl              shared definitions (map, Jacobian, P, F, N, bilinear form B, coefficients)
  01_fixed_points.wl       Sections 2-3: symmetry, z-axis, fixed points, characteristic polynomials
  02_stability.wl          Section 4:   disc criterion, polynomials F and N, Routh-Hurwitz, Remark 2, thresholds
  03_bifurcation.wl        Section 5:   transversality, closed form of l_0, l_F, d, rho_sw, rho_G
  04_chaos_control.wl      Section 6:   controlled characteristic polynomial, p(1), -p(-1), Example 7.5
  run_all.wls              runs 01-04 in order
  output/                  reference output of each script
python/
  lorenz_map.py            shared definitions (map, Jacobian, thresholds, normal form coefficients, Jury test,
                           vectorized bifurcation/Lyapunov scan)
  report.py                prints results and writes output/*.txt with PASS/FAIL checks
  01_fixed_points_and_stability.py   Sections 2-4 and the critical values of Section 7.1
  02_bifurcation_coefficients.py     Section 5 and Section 7.1 (signs of l_0, l_F, d; rho_G; resonances)
  03_numerical_examples.py           every number quoted in Examples 7.1-7.4
  04_control.py                      Section 6 and Example 7.5
  05_make_figures.py                 regenerates the nine figures of Section 7 into figures/
  figures_src/                       plotting code used by 05 (common style, colour-blind safe palette)
  run_all.py                         runs 01-05 in order
  output/                  reference output of each script
figures/                   the nine figures of the paper (PDF, with PNG previews)
```

## Where each result of the paper is checked

| Paper | Mathematica | Python |
|---|---|---|
| Lemmas 2.1–2.2, Theorem 3.1, Lemma 3.1, eq. (3.3) | `01_fixed_points.wl` | `01_fixed_points_and_stability.py` |
| Lemma 4.1, Theorems 4.1–4.2, Lemma 4.2, Remark 2 | `02_stability.wl` | `01_fixed_points_and_stability.py` |
| Lemma 5.1, Theorems 5.1–5.3 (coefficients l_0, l_F, d) | `03_bifurcation.wl` | `02_bifurcation_coefficients.py` |
| Section 7.1: rho_1, rho_sw, rho_G, signs of the coefficients, no strong resonance | `02_stability.wl`, `03_bifurcation.wl` | `01_…`, `02_bifurcation_coefficients.py` |
| Examples 7.1–7.4 | – | `03_numerical_examples.py` |
| Theorem 6.1 and Example 7.5 | `04_chaos_control.wl` | `04_control.py` |
| Figures 1–9 | – | `05_make_figures.py` |

## How to run

**Python** (3.9 or later):

```bash
cd python
pip install -r requirements.txt
python run_all.py            # or run 01 ... 05 one at a time
```

Each script prints its checks and stops with an error if any of them fails. A full run takes about
two minutes on a laptop; the reference output is in `python/output/`.

**Mathematica** (version 12 or later, or the free Wolfram Engine):

```bash
cd mathematica
wolframscript -file run_all.wls
```

Each `.wl` file can also be opened in a notebook and evaluated on its own; it loads `00_model.wl` itself.

## Verification summary

All 77 checks pass: 30 in Mathematica and 47 in Python.

* **Symbolic (Mathematica).** These hold for all parameter values:
  * the symmetry of the map;
  * the fixed points;
  * both characteristic polynomials and the relation `det(mu I - J) = h^3 P(lambda)`;
  * the disc criterion;
  * the identity `h^3 P(-2/h) = 2F(h)`;
  * the factorization behind Lemma 4.2;
  * `N(0) = 2c - KB`, which vanishes exactly at `rho_H`;
  * the closed form of `l_0`;
  * the controlled characteristic polynomial, `p(1)` and `-p(-1)`.
* **Thresholds.** For nine parameter sets, the thresholds obtained from `F` and `N` agree with the
  minimum of `h(lambda)` over the numerically computed eigenvalues. The spectral radius is below 1 at
  `0.999 h_E` and above 1 at `1.001 h_E`.
* **Normal form coefficients.** These are computed independently in Mathematica and in Python, and the
  values agree to all printed digits.
* **Coefficient signs (sigma = 10, beta = 8/3).** `l_0 < 0` on (0, 1); `l_F` lies in [0.0065, 0.0108]
  on (1, rho_sw); `d < 0` on (rho_sw, rho_G) and `d > 0` on (rho_G, rho_H), with
  `rho_G = 8.6219`. Also `max h_NS^2 m = 0.362 < 2`, so no strong resonance occurs.
* **Directions of the bifurcations.** These are confirmed by direct iteration:
  * the bifurcating 2-cycles and invariant curves grow like `sqrt(|h - h_c|)`: their size doubles when
    the distance to the threshold is multiplied by four;
  * the unstable symmetric 2-cycle of the origin has multiplier modulus 1.008;
  * the subcritical Neimark–Sacker bifurcation at rho = 10 jumps to a distant chaotic attractor.
* **Jury region.** The stability region from the Jury conditions agrees with a direct eigenvalue test
  on a 301 × 401 grid of gains. The only exceptions are grid points that lie exactly on the boundary,
  where the spectral radius equals 1 to machine precision.
* **Reproducibility of chaotic orbits.** The map is evaluated with exactly the same floating-point
  expression in all scripts. Round-off is amplified on a chaotic attractor, so quantities such as the
  switch-on times n = 3009 and n = 3717 of Example 7.5 are reproduced iterate by iterate. The
  Lyapunov exponent at h = 0.07 is averaged over 10^6 iterates from three initial points
  (0.0426–0.0428 per iteration).

## Licence

MIT, see `LICENSE`. If you use this code, please cite the paper above.
