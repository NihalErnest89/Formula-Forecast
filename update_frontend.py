"""
One-command pipeline to bring the GitHub Pages frontend up to date.

Steps:
  1. collect_data.py          - incremental: only fetches races newer than the
                                data/raw/ snapshots (seconds if nothing new)
  2. top10/train.py           - retrain the model on the refreshed data
  3. generate_static_data.py  - precompute races.json + per-race prediction JSONs
  4. npm run deploy           - build the React app and push to the gh-pages branch

Usage:
    python update_frontend.py                 # full pipeline incl. deploy
    python update_frontend.py --skip-train    # reuse the existing model
    python update_frontend.py --skip-deploy   # refresh data/predictions only
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def run_step(name, cmd, cwd=None, shell=False):
    print(f"\n{'=' * 60}\nSTEP: {name}\n{'=' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd or ROOT, shell=shell)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode}) after {elapsed:.0f}s")
        sys.exit(result.returncode)
    print(f"OK: {name} ({elapsed:.0f}s)")


def main():
    skip_train = '--skip-train' in sys.argv
    skip_deploy = '--skip-deploy' in sys.argv

    run_step("Collect new race data (incremental)", [sys.executable, 'collect_data.py'])

    if skip_train:
        print("\nSkipping training (--skip-train): reusing existing model")
    else:
        run_step("Retrain model", [sys.executable, str(ROOT / 'top10' / 'train.py')])

    run_step("Generate static prediction data", [sys.executable, 'generate_static_data.py'])

    if skip_deploy:
        print("\nSkipping deploy (--skip-deploy). To publish:")
        print("  cd frontend && npm run deploy")
    else:
        # npm is npm.cmd on Windows, so run through the shell
        run_step("Build and deploy to GitHub Pages", 'npm run deploy',
                 cwd=ROOT / 'frontend', shell=True)
        print("\nDeployed: https://NihalErnest89.github.io/Formula-Forecast")


if __name__ == "__main__":
    main()
