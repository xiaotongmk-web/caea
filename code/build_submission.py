"""Build anonymous sources from public text and the official-style base."""
from pathlib import Path
import re,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'paper';d=ROOT/'submission'
# The stable base contains split two-column equations and unmodified official style.
base=d/'main_base_v02.tex'
if not base.exists():shutil.copy(d/'main.tex',base)
s=(p/'main.tex').read_text();b=base.read_text()
abstract=re.search(r'\\begin\{abstract\}.*?\\end\{abstract\}',s,re.S).group()
b=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:abstract,b,flags=re.S)
a=b.index('\\subsection{Frozen-backbone candidate selection}');z=b.index('\\section{Discussion and limitations}',a)
b=b[:a]+r'''\subsection{Fresh multi-model candidate scoring}
\label{sec:scale}
'''+(p/'scale_main_section.tex').read_text()+r'''
\paragraph{Original head-DPO pilot.}
The earlier frozen 135M/0.5B head-DPO pilot used 40/12/24 contexts and five
label seeds. Apparent gains failed an answer-permutation control. Those
archived results and paired intervals remain in the supplement, distinct
from the fresh inference above.

'''+b[z:]
for block in re.findall(r'\\begin\{table\*\}.*?\\end\{table\*\}',b,re.S):
 if 'table_review_uncertainty.tex'in block:b=b.replace(block,'')
for block in re.findall(r'\\begin\{table\}\[t\].*?\\end\{table\}',b,re.S):
 bb=block.replace('\\begin{table}','\\begin{table*}').replace('\\end{table}','\\end{table*}').replace('\\linewidth','\\textwidth')
 b=b.replace(block,bb)
discussion=r'''\section{Discussion and limitations}
Stability does not imply correctness. The synthetic oracle targets isolate
repair, while the expanded study fits only two calibration parameters of a
frozen auditor. Constant targets explain most apparent gains; more samples
cannot remove systematic audit bias. Full repair and audit-only share risk.
Spread increases report variance. Conditional preservation can retain
position shortcuts and need not retain aggregate style.

Report classes must capture the signal driving the specified update;
unmeasured persuasive wording invalidates that representation. The
mean-response bifurcation is not a zero-variance or nonergodicity claim for
the sampled process. Global regularity and evidence coverage have not been
established for unrestricted language models.

Released checkpoints, quantization, training and interface differences
confound causal parameter-scaling or RLHF claims. Candidate mass and order
failures are reported without exclusions. The external task is a small,
length-filtered correctness subset, possibly exposed in pretraining, not
standard GSM8K solve accuracy. Bootstrap intervals exclude checkpoint and
natural-conversation variation. Unseen free-text responses, independently
adjudicated open-domain audits and human belief effects remain untested;
human effects require direct measurement and applicable ethical review.

'''
b=re.sub(r'\\section\{Discussion and limitations\}.*?(?=\\section\{Conclusion\})',lambda _:discussion,b,flags=re.S)
b=b.replace('a different objective from matching a justified report target.','a different objective from matching a justified report target.\nSYCON studies free-form multi-turn sycophancy \\citep{hong2025}; our finite\nresponse banks do not substitute for that setting.')
b=b.replace('the frozen-neural pilot fails an option-permutation\ncontrol.','the original neural pilot fails an option-permutation\ncontrol. A larger fresh study calibrates its auditor without test labels\nand compares three checkpoints up to 7.61B parameters against shrinkage.')
b=b.replace('a different audit specification and are outside these experiments.','a different audit specification and are outside these experiments.\nThe expanded non-oracle study substitutes a separately calibrated\n$\\widehat p_x$; test labels evaluate but never construct those targets.')
(d/'main.tex').write_text(b)
macros=s[s.index('\\newtheorem{theorem}'):s.index('\\title{')]
proof=s.split('\\appendix',1)[1].rsplit('\\end{document}',1)[0]
pre=r'''\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{microtype,amsmath,amssymb,amsthm,booktabs,graphicx,array}
\usepackage[hidelinks]{hyperref}
\usepackage{xr}
\externaldocument{main}
\newcommand{\citep}[1]{}
'''+macros+r'''\title{CAEA: Supplementary Proofs and Reproduction Details}
\author{}\date{}
\begin{document}\maketitle\appendix
'''
extra=r'''\clearpage
\section{Original frozen-head pilot and paired uncertainty}
'''+(p/'neural_section.tex').read_text()+r'''\clearpage
\section{Expanded-study design and diagnostics}
'''+(p/'scale_section.tex').read_text()+'\n\\end{document}\n'
extra=extra.replace('tab:scale','tab:suppscale').replace('tab:auditor','tab:suppauditor')
(d/'appendix.tex').write_text(pre+proof+extra)
for name in ['references.bib','table_results.tex','table_neural.tex','table_review_uncertainty.tex','table_scale.tex','table_auditor.tex']:
 shutil.copy(p/name,d/name)
shutil.copytree(p/'figures',d/'figures',dirs_exist_ok=True)
if '--compile'in sys.argv:
 for name in ['main','appendix']:
  commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']]
  if name=='main':commands.append(['bibtex',name])
  commands.extend([['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']]*2)
  for cmd in commands:subprocess.run(cmd,cwd=d,check=True)
print('Anonymous sources prepared with unmodified official style files.')
