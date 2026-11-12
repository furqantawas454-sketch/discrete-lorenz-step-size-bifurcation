"""Runs scripts 01-05 in the order of the paper."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
for script in ["01_fixed_points_and_stability.py", "02_bifurcation_coefficients.py",
               "03_numerical_examples.py", "04_control.py", "05_make_figures.py"]:
    print("\n" + "=" * 78 + f"\n{script}\n" + "=" * 78, flush=True)
    subprocess.run([sys.executable, script], cwd=here, check=True)
