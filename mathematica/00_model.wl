(* ::Package:: *)
(* 00_model.wl -- shared definitions for the Euler-discretized Lorenz system.
   Symbols: s = sigma, b = beta, r = rho, h = step size.  Equation numbers refer to the manuscript. *)

ClearAll["Global`*"];
Fvec[{x_, y_, z_}, s_, b_, r_] := {s (y - x), x (r - z) - y, x y - b z};
Gmap[{x_, y_, z_}, h_, s_, b_, r_] := {x + h s (y - x), y + h (x (r - z) - y), z + h (x y - b z)};   (* (2.4) *)
Jac[{x_, y_, z_}, h_, s_, b_, r_] := {{1 - h s, h s, 0}, {h (r - z), 1 - h, -h x}, {h y, h x, 1 - h b}}; (* (3.2) *)
DFm[{x_, y_, z_}, s_, b_, r_] := {{-s, s, 0}, {r - z, -1, -x}, {y, x, -b}};
Ep[b_, r_] := {Sqrt[b (r - 1)], Sqrt[b (r - 1)], r - 1};                                             (* (3.1) *)
Sm = DiagonalMatrix[{-1, -1, 1}];
Ppoly[l_, s_, b_, r_] := l^3 + (s + b + 1) l^2 + b (s + r) l + 2 s b (r - 1);                     (* (3.5) *)
rhoH[s_, b_] := s (s + b + 3)/(s - b - 1);                                                          (* (2.3) *)
Fpoly[h_, s_, b_, r_] := s b (r - 1) h^3 - b (s + r) h^2 + 2 (s + b + 1) h - 4;                     (* (4.5) *)
mfun[h_, s_, b_, r_] := b (s + r) - 2 s b (r - 1) h;
Npoly[h_, s_, b_, r_] := mfun[h, s, b, r] (h mfun[h, s, b, r] - (s + b + 1)) + 2 s b (r - 1);        (* (4.6) *)
Bform[u_, v_, h_] := h {0, -(u[[1]] v[[3]] + u[[3]] v[[1]]), u[[1]] v[[2]] + u[[2]] v[[1]]};        (* (5.2) *)

(* numerical thresholds for E_+ (Lemma 4.2, Theorem 4.2) *)
hFnum[s_, b_, r_?NumericQ] := Min[Select[h /. NSolve[Fpoly[h, s, b, N[r]] == 0, h, Reals], # > 0 &]];
hNSnum[s_, b_, r_?NumericQ] := Module[{c = Select[h /. NSolve[Npoly[h, s, b, N[r]] == 0, h, Reals],
      # > 0 && mfun[#, s, b, r] > 0 && #^2 mfun[#, s, b, r] < 4 &]}, If[c === {}, Infinity, Min[c]]];

(* normal form coefficients (Theorems 5.2, 5.3); |q| = 1 and <p,q> = 1 *)
flipCoef[X_, h_, s_, b_, r_] := Module[{A = N[Jac[X, h, s, b, r]], ev, evl, q, p},
   ev = Eigensystem[A]; q = Re[ev[[2, First@Ordering[Abs[ev[[1]] + 1], 1]]]]; q = q/Norm[q];
   evl = Eigensystem[Transpose[A]]; p = Re[evl[[2, First@Ordering[Abs[evl[[1]] + 1], 1]]]]; p = p/(p . q);
   -(1/2) p . Bform[q, Inverse[A - IdentityMatrix[3]] . Bform[q, q, h], h]];
nsCoef[s_, b_, r_?NumericQ] := Module[{h = hNSnum[s, b, r], A, ev, evl, i1, mu, q, p, th, w},
   A = N[Jac[Ep[b, r], h, s, b, r]]; ev = Eigensystem[A]; i1 = First@Ordering[-Im[ev[[1]]], 1];
   mu = ev[[1, i1]]; q = ev[[2, i1]]; q = q/Norm[q];
   evl = Eigensystem[Transpose[A]]; p = evl[[2, First@Ordering[Abs[evl[[1]] - Conjugate[mu]], 1]]];
   p = p/Conjugate[Conjugate[p] . q]; th = Arg[mu];
   w = 2 Conjugate[p] . Bform[q, Inverse[IdentityMatrix[3] - A] . Bform[q, Conjugate[q], h], h] +
       Conjugate[p] . Bform[Conjugate[q], Inverse[Exp[2 I th] IdentityMatrix[3] - A] . Bform[q, q, h], h];
   {(1/2) Re[Exp[-I th] w], h, th}];

(* state feedback (Section 6) *)
Jcontrolled[k1_, k2_, h_, s_, b_, r_] := Jac[Ep[b, r], h, s, b, r] - {{k1, k2, 0}, {0, 0, 0}, {0, 0, 0}};
bisect[f_, {a0_, b0_}, n_] := Module[{a = a0, b = b0, m}, Do[m = (a + b)/2; If[Sign[f[m]] == Sign[f[a]], a = m, b = m], {n}]; (a + b)/2];
check[label_, test_] := Print[If[TrueQ[test], "    [PASS] ", "    [FAIL] "], label];
