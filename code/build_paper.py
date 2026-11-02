"""Copy regenerated figures and compile the public manuscript."""
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    paper=ROOT/'paper'
    for source in (ROOT/'figures').glob('*.pdf'):
        if 'singlecolumn' not in source.name:
            shutil.copy2(source,paper/'figures'/source.name)
    commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
              ['bibtex','main'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]
    for command in commands:
        subprocess.run(command,cwd=paper,check=True)
    shutil.copy2(paper/'main.pdf',paper/'CAEA.pdf')
    print(paper/'CAEA.pdf')


if __name__=='__main__':main()
