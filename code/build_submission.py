"""Compile the historical IJCAI draft bundled with the public repository."""
import argparse
from pathlib import Path
import subprocess

SUBMISSION = Path(__file__).resolve().parents[1] / "submission"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Run pdflatex and bibtex")
    if parser.parse_args().compile:
        for name in ("main", "appendix"):
            latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"]
            subprocess.run(latex, cwd=SUBMISSION, check=True)
            if name == "main":
                subprocess.run(["bibtex", name], cwd=SUBMISSION, check=True)
            for _ in range(2):
                subprocess.run(latex, cwd=SUBMISSION, check=True)
    print(f"Historical IJCAI sources: {SUBMISSION}")


if __name__ == "__main__":
    main()
