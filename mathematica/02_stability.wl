(* 02_stability.wl -- Section 4: disc criterion, thresholds F and N, Routh-Hurwitz, Remark 2 *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "00_model.wl"}]];
Print["== Section 4 =="];
check["Lemma 4.1: |1 + h(a+ib)|^2 - 1 = h(2a + h(a^2+b^2))", Simplify[ComplexExpand[Abs[1 + h (a + I bb)]^2] - 1 - h (2 a + h (a^2 + bb^2))] == 0];
check["Lemma 4.2(i): h^3 P(-2/h) = 2 F(h)", Simplify[h^3 Ppoly[-2/h, s, b, r] - 2 Fpoly[h, s, b, r]] == 0];
(* Lemma 4.2(ii): with a = -h m/2, l1 = -K + h m, m = m(h), the cubic (l - l1)(l^2 - 2a l + m) equals P(l) iff N(h) = 0 *)
K = s + b + 1; c = s b (r - 1); Bc = b (s + r); m = mfun[h, s, b, r];
diffpoly = Expand[(l - (-K + h m)) (l^2 + h m l + m) - Ppoly[l, s, b, r]];
check["Lemma 4.2(ii): (l - l1)(l^2 - 2a l + m) - P(l) = -N(h)(1 + h l), so it equals P exactly when N(h) = 0", Simplify[diffpoly + Npoly[h, s, b, r] (1 + h l)] == 0];
check["Routh-Hurwitz: K B - 2c = b[s(s+b+3) - r(s-b-1)]", Simplify[K Bc - 2 c - b (s (s + b + 3) - r (s - b - 1))] == 0];
check["Remark 2: N(0) = 2c - K B, zero exactly at rho_H", Simplify[Npoly[0, s, b, r] - (2 c - K Bc)] == 0 && Simplify[Npoly[0, s, b, rhoH[s, b]]] == 0];
check["rho_H(10, 8/3) = 470/19", rhoH[10, 8/3] == 470/19];
Print["  thresholds for sigma = 10, beta = 8/3:"];
Do[With[{hf = hFnum[10, 8/3, rr], hn = hNSnum[10, 8/3, rr]},
   Print["    rho = ", rr, ":  h_F = ", NumberForm[N[hf], 6], ",  h_NS = ", If[hn === Infinity, Infinity, NumberForm[N[hn], 6]]]],
  {rr, {1.2, 2., 5., 10., 15., 20., 24.}}];
check["h_F(2) = 0.17790, h_NS(5) = 0.10282, h_NS(10) = 0.03096",
  Abs[hFnum[10, 8/3, 2.] - 0.17790049] < 10^-7 && Abs[hNSnum[10, 8/3, 5.] - 0.10282039] < 10^-7 && Abs[hNSnum[10, 8/3, 10.] - 0.03095511] < 10^-7];
rho1 = r /. FindRoot[Discriminant[Ppoly[l, 10, 8/3, r], l] == 0, {r, 1.3}, WorkingPrecision -> 30];
Print["  rho_1 = ", NumberForm[rho1, 6]];
check["rho_1 = 1.3456 (complex pair appears)", Abs[rho1 - 1.34562] < 10^-4];
h0[rr_] := Min[4/(10 + 1 + Sqrt[81 + 40 rr]), 3/4];
check["Theorem 4.1, rho = 0.5: h_0 = h_1 = 0.19002", Abs[N[h0[1/2]] - 0.190025] < 10^-6];
