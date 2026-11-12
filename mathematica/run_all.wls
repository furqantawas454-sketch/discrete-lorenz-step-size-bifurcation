#!/usr/bin/env wolframscript
(* Runs 01-04 in the order of the paper:  wolframscript -file run_all.wls *)
dir = DirectoryName[$InputFileName];
Do[Print["\n==================== ", f, " ===================="]; Get[FileNameJoin[{dir, f}]],
  {f, {"01_fixed_points.wl", "02_stability.wl", "03_bifurcation.wl", "04_chaos_control.wl"}}];
