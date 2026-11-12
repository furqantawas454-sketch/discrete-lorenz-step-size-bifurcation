"""05 -- regenerates all nine figures of Section 7 into ../figures (PDF, plus PNG previews)."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "figures_src")
for script in ["fig_overview.py", "fig_examples.py", "fig_control.py"]:
    print(f"running {script} ...", flush=True)
    subprocess.run([sys.executable, script], cwd=src, check=True)
figdir = os.path.join(here, "..", "figures")
try:
    for f in sorted(os.listdir(figdir)):
        if f.endswith(".pdf"):
            base = os.path.join(figdir, f[:-4])
            subprocess.run(["pdftoppm", "-r", "110", "-png", "-singlefile", base + ".pdf", base], check=True)
except FileNotFoundError:
    print("pdftoppm not found: PNG previews skipped (the PDFs are complete).")
print("figures written to", os.path.abspath(figdir))
