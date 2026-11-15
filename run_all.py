"""Run every experiment of the assignment and regenerate results/.

    python run_all.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
SCRIPTS = ["part1_second_price.py", "part2_first_price.py", "part3_vcg_path.py"]

for script in SCRIPTS:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, script], cwd=SRC, check=True)
print("\nAll figures are in results/figures and all numbers in results/data.")
