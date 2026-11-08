"""Replay archived CPU checks in a disposable copy; never invoke inference."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    ["code/verify_results.py"],
    ["code/review_audit.py"],
    ["code/verify_scale.py"],
    ["code/verify_external.py"],
    ["-m", "unittest", "discover", "-s", "code", "-p", "test_research_integrity.py", "-v"],
    ["code/evidence_audit_diagnostics.py", "--out", "diagnostics"],
]


def fingerprints(root):
    records = {}
    for name in ("code", "results"):
        for path in sorted((root / name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                records[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New report directory")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds per check")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    # Some historical verifiers write fixed filenames. Never run them in source.
    if ROOT == args.out.resolve() or ROOT in args.out.resolve().parents:
        parser.error("Report directory must be outside the repository")
    before = fingerprints(ROOT)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(python=sys.version, platform=platform.platform(), input_sha256=before,
                  scope="Archived CPU consistency/replay, not fresh inference or server DPO verification", checks=[])
    report["packages"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}
    with tempfile.TemporaryDirectory(prefix="caea-replay-") as directory:
        work = Path(directory)
        for name in ("code", "results"):
            shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__"))
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLCONFIGDIR=str(work / "mpl"))
        for i, check in enumerate(CHECKS):
            command = [sys.executable] + check
            try:
                result = subprocess.run(command, cwd=work, env=env, capture_output=True,
                                        text=True, timeout=args.timeout)
                code, log = result.returncode, result.stdout + result.stderr
            except subprocess.TimeoutExpired as exc:
                code, log = 124, "Timed out: " + str(exc)
            (args.out / (str(i) + ".log")).write_text(log, encoding="utf-8")
            report["checks"].append(dict(command=check, returncode=code, log=str(i) + ".log"))
            print("PASS" if code == 0 else "FAIL", " ".join(check), flush=True)
    report["source_unchanged"] = before == fingerprints(ROOT)
    report["passed"] = report["source_unchanged"] and all(c["returncode"] == 0 for c in report["checks"])
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
