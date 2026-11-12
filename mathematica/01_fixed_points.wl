(* 01_fixed_points.wl -- Sections 2-3: symmetry, z-axis, fixed points, characteristic polynomials *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "00_model.wl"}]];
Print["== Sections 2-3 =="];
check["Lemma 2.1: G_h(SX) = S G_h(X)", Simplify[Gmap[Sm . {x, y, z}, h, s, b, r] - Sm . Gmap[{x, y, z}, h, s, b, r]] == {0, 0, 0}];
check["Lemma 2.2: z-axis invariant, z -> (1 - h b) z", Simplify[Gmap[{0, 0, z}, h, s, b, r] - {0, 0, (1 - h b) z}] == {0, 0, 0}];
fp = Solve[Gmap[{x, y, z}, h, s, b, r] == {x, y, z} && h != 0 && s != 0, {x, y, z}];
Print["  fixed points: ", {x, y, z} /. fp];
check["Theorem 3.1: exactly E_0, E_+ and E_-", Sort[Simplify[{x, y, z} /. fp]] ===
   Sort[Simplify[{{0, 0, 0}, {Sqrt[b] Sqrt[r - 1], Sqrt[b] Sqrt[r - 1], r - 1}, {-Sqrt[b] Sqrt[r - 1], -Sqrt[b] Sqrt[r - 1], r - 1}}]]];
check["Eq. (3.3): J = I + h DF", Simplify[Jac[{x, y, z}, h, s, b, r] - (IdentityMatrix[3] + h DFm[{x, y, z}, s, b, r])] == ConstantArray[0, {3, 3}]];
cp0 = Factor[CharacteristicPolynomial[DFm[{0, 0, 0}, s, b, r], l]];
check["Lemma 3.1(i): char. polynomial at E_0 = -(l + b)(l^2 + (s+1) l + s(1-r))", Simplify[cp0 + (l + b) (l^2 + (s + 1) l + s (1 - r))] == 0];
cpE = Simplify[-CharacteristicPolynomial[DFm[Ep[b, r], s, b, r], l], r > 1 && b > 0];
check["Lemma 3.1(ii): char. polynomial at E_+ is P, eq. (3.5)", Simplify[cpE - Ppoly[l, s, b, r]] == 0];
cpmap = Simplify[-CharacteristicPolynomial[Jac[Ep[b, r], h, s, b, r], mu] /. mu -> 1 + h l, r > 1 && b > 0];
check["det(mu I - J(E_+)) = h^3 P(lambda) with mu = 1 + h lambda", Simplify[cpmap - h^3 Ppoly[l, s, b, r]] == 0];
