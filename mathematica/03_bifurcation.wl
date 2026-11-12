(* 03_bifurcation.wl -- Section 5 and Section 7.1: transversality, l_0, l_F, d, rho_sw, rho_G *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "00_model.wl"}]];
Print["== Section 5 =="];
check["Lemma 5.1(ii): d|mu|^2/dh at h = -2a/(a^2+b^2) equals -2a", Simplify[(D[1 + 2 h a + h^2 (a^2 + bb^2), h] /. h -> -2 a/(a^2 + bb^2)) + 2 a] == 0];
(* Theorem 5.1: closed form of l_0 *)
lm = (-(s + 1) - Sqrt[(s - 1)^2 + 4 s r])/2; h1 = 2/Abs[lm];
q = {1, (s + lm)/s, 0}; p = {r/(s + lm), 1, 0}; p = p/(p . q);
A0 = Jac[{0, 0, 0}, h1, s, b, r];
l0gen = -(1/2) p . Bform[q, Inverse[A0 - IdentityMatrix[3]] . Bform[q, q, h1], h1];
l0closed = -(h1/b) (s + lm)^2/(s r + (s + lm)^2);
check["Theorem 5.1: general formula reduces to the closed form (5.4)", Simplify[l0gen - l0closed, s > 0 && b > 0 && r > 0] == 0];
Print["  l_0(rho = 0.5) = ", N[l0closed /. {s -> 10, b -> 8/3, r -> 1/2}]];
check["l_0(0.5) = -0.003722 < 0", Abs[N[l0closed /. {s -> 10, b -> 8/3, r -> 1/2}] + 0.0037221] < 10^-6];
Print["  l_F: ", Table[{rr, N[flipCoef[Ep[8/3, rr], hFnum[10, 8/3, rr], 10, 8/3, rr]]}, {rr, {1.2, 2., 3.}}]];
check["Theorem 5.2: l_F(2) = 0.008676 > 0", Abs[flipCoef[Ep[8/3, 2.], hFnum[10, 8/3, 2.], 10, 8/3, 2.] - 0.0086757] < 10^-6];
ns5 = nsCoef[10, 8/3, 5.]; ns10 = nsCoef[10, 8/3, 10.];
Print["  rho = 5:  d = ", N[ns5[[1]]], ", h_NS = ", N[ns5[[2]]], ", theta_0 = ", N[ns5[[3]]]];
Print["  rho = 10: d = ", N[ns10[[1]]], ", h_NS = ", N[ns10[[2]]], ", theta_0 = ", N[ns10[[3]]]];
check["Theorem 5.3: d(5) = -1.6278e-3 (supercritical), d(10) = 3.7923e-5 (subcritical)", Abs[ns5[[1]] + 0.0016278] < 10^-7 && Abs[ns10[[1]] - 0.000037923] < 10^-9];
rsw = rr /. FindRoot[hFnum[10, 8/3, rr] - hNSnum[10, 8/3, rr], {rr, 3.4, 3.8}];
rG = bisect[nsCoef[10, 8/3, #][[1]] &, {7., 10.}, 40];
Print["  rho_sw = ", rsw, " (h = ", N[hFnum[10, 8/3, rsw]], "),  rho_G = ", rG, " (h = ", N[hNSnum[10, 8/3, rG]], ")"];
check["rho_sw = 3.6288, rho_G = 8.6219", Abs[rsw - 3.62879] < 10^-4 && Abs[rG - 8.62187] < 10^-4];
