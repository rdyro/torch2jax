#!/usr/bin/env python3
"""Extract python code blocks from README/docs and check that each one runs."""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = "import torch\nimport jax\nimport jax.numpy as jnp\nfrom torch2jax import torch2jax, torch2jax_without_vjp\n\n"


def extract_python_blocks(path: Path) -> list[str]:
    pattern = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
    return pattern.findall(path.read_text(encoding="utf-8"))


def main():
    target_dir = ROOT / "temp_examples_from_docs"
    target_dir.mkdir(exist_ok=True)
    for stale in target_dir.glob("*.py"):
        stale.unlink()

    files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    scripts = []
    for path in files:
        if not path.exists():
            continue
        for block in extract_python_blocks(path):
            if not block.strip():
                continue
            safe_name = path.relative_to(ROOT).as_posix().replace("/", "_").replace(".md", "")
            script = target_dir / f"{safe_name}_example_{len(scripts) + 1}.py"
            script.write_text(HEADER + block, encoding="utf-8")
            scripts.append(script)

    print(f"Extracted {len(scripts)} examples to {target_dir}")
    print("\nChecking if they run (timeout=60s per script)...")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    passed, failed = [], []
    for script in scripts:
        print(f"Running {script}...")
        try:
            result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60, env=env)
        except subprocess.TimeoutExpired:
            print("  [TIMEOUT]")
            failed.append(script)
            continue
        if result.returncode == 0:
            print("  [SUCCESS]")
            passed.append(script)
        else:
            tail = "\n".join(result.stderr.strip().splitlines()[-5:])
            print(f"  [FAILED] exit {result.returncode}\n{tail}")
            failed.append(script)

    print(f"\n--- Summary ---\nPassed: {len(passed)}\nFailed: {len(failed)}")
    for script in failed:
        print(f"  - {script}")


if __name__ == "__main__":
    main()
