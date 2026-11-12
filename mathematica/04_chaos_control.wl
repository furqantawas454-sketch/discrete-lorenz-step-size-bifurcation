(* 04_chaos_control.wl -- Section 6 and Example 7.5: controlled characteristic polynomial and Jury conditions *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "00_model.wl"}]];
Print["== Section 6 =="];
cp = CoefficientList[Simplify[-CharacteristicPolynomial[Jcontrolled[k1, k2, h, s, b, r], mu], r > 1 && b > 0], mu];
{a0, a1, a2} = Simplify[cp[[1 ;; 3]]];
check["leading coefficient 1", Simplify[cp[[4]]] == 1];
check["a2 of (6.2)", Simplify[a2 - (k1 - 3 + h (s + b + 1))] == 0];
check["a1 of (6.2)", Simplify[a1 - (3 - 2 k1 + h ((b + 1) k1 + k2 - 2 (s + b + 1)) + b h^2 (s + r))] == 0];
check["a0 of (6.2)", Simplify[a0 - (k1 - 1 + h ((s + b + 1) - (b + 1) k1 - k2) - b h^2 ((s + r) - r k1 + (r - 2) k2) + 2 s b (r - 1) h^3)] == 0];
check["p(1) of (6.3)", Simplify[1 + a2 + a1 + a0 - b h^2 (r k1 + (2 - r) k2 + 2 s (r - 1) h)] == 0];
check["-p(-1) of (6.4)", Simplify[-(-1 + a2 - a1 + a0) - (8 - 4 k1 - 4 h (s + b + 1) + 2 h (b + 1) k1 + 2 h k2 + b h^2 (2 (s + r) - r k1 + (r - 2) k2) - 2 s b (r - 1) h^3)] == 0];
JcM = Jac[Sm . Ep[b, r], h, s, b, r] - {{k1, k2, 0}, {0, 0, 0}, {0, 0, 0}};
check["the same gains at E_-: J_c(E_-) = S J_c(E_+) S", Simplify[JcM - Sm . Jcontrolled[k1, k2, h, s, b, r] . Sm] == ConstantArray[0, {3, 3}]];
rad[k1v_, k2v_, hv_, rv_] := Max[Abs[Eigenvalues[N[Jcontrolled[k1v, k2v, hv, 10, 8/3, rv]]]]];
Print["  (rho,h) = (10,0.07): spectral radius ", rad[0, 0, 7/100, 10], " -> ", rad[1/2, 1/2, 7/100, 10]];
Print["  (rho,h) = (28,0.01): spectral radius ", rad[0, 0, 1/100, 28], " -> ", rad[1/2, 0, 1/100, 28]];
check["Example 7.5: 1.0513 -> 0.9575 and 1.0061 -> 0.9877",
  Abs[rad[0, 0, 7/100, 10] - 1.0513] < 10^-4 && Abs[rad[1/2, 1/2, 7/100, 10] - 0.9575] < 10^-4 &&
  Abs[rad[0, 0, 1/100, 28] - 1.0061] < 10^-4 && Abs[rad[1/2, 0, 1/100, 28] - 0.9877] < 10^-4];
