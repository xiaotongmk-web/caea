# CAEA: Auditing and Repairing Preference-Induced Belief Feedback

**Xiaotong Zhang - China University of Petroleum (East China)**

[Manuscript](paper/CAEA.pdf) | [Repository](https://github.com/xiaotongmk-web/caea) | [Citation](CITATION.cff) | [Recorded data](results/)

This unpublished theory and controlled computational study has no assigned arXiv
identifier or DOI. The neural pilots use frozen backbones and trained candidate
heads. There is no full-model post-training, free-text evaluation or human study.

## Main finding

Pooling over stance can hide feedback sensitivity. A binary construction has
the same pooled response/outcome law under both contracting and separated-
equilibrium mean-response dynamics. An extension has arbitrarily small report
regret, with rescaled agreement pressure and small absolute belief separation.
CAEA replaces an externally supplied report-class marginal while retaining the
preference policy within each class. The KL projection is classical. Full repair
and audit-only necessarily have identical report risk; partial repair's risk and
target-TV changes follow algebraically. Experiments examine implementation,
conditional preferences, cost and failure cases, rather than independent
discovery of an evidence target.

![Controlled audit observability example](figures/audit_concept.png)

## Small example

Use Python 3.12 and the recorded core versions:

```bash
python -m venv .venv
python -m pip install -r requirements-core.txt
python code/demo.py --pressure 4 --repair 0
python code/demo.py --pressure 4 --repair 0.6
```

Activate before installing: `source .venv/bin/activate` on POSIX shells or
`.\.venv\Scripts\Activate.ps1` on Windows PowerShell. The demo prints two
deterministic trajectories; it does not measure human behavior.

## Offline release verification

```bash
python code/verify_results.py
python code/review_audit.py
```

The first checks study consistency. The second independently replays archived
frozen-model hidden states/logits and fitted heads, checks repair identities and
empty-support behavior, and reconstructs paired bootstrap intervals over 24
evidence clusters and five label seeds. It does not perform fresh backbone
inference or independently authenticate the archived logits. Up to `1e-7` replay
roundoff is allowed for archived float32 PCA features. The actual error and
uncertainty scope are recorded in `results/review_verification.json`.

## Reproduce the controlled study

Run sequentially from the project root:

```bash
python code/experiment.py
python code/stress.py
python code/audit_stability.py
python code/review_audit.py
python code/report.py
python code/plot_publication.py
python code/plot_concept.py
python code/verify_results.py
python code/build_paper.py
```

Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before starting Python for
comparable CPU execution. Runners overwrite named results, so preserve a copy
first. Main and stress runners both write deterministic phase outputs and must
run sequentially. The build script copies regenerated figures into the paper
folder and invokes pdflatex/bibtex.

| Component | Setting |
| --- | --- |
| Main study | 10 seeds, 24,000 training / 4,000 held-out pairs per seed, 63 responses, 14 methods |
| Main evaluation | 600 contexts x 9 stances per seed |
| Extra pressure fits | 20 fits, 16,000 pairs each, 3 temperatures, 3 stance groups |
| Stochastic feedback | 5,000 trajectories, 400 rounds, 10 seeds, 2 gains, 5 repair fractions |
| Finite response audits | 5 budgets, 500 repetitions each, supplied curvature, `H*h/2` remainder |
| Frozen-neural pilot | 40/12/24 train/validation/test contexts, 152 prompts/model, 5 label seeds |
| Transfer control | 48 permuted prompts/model; no robust DPO gain established |

## Fresh frozen-model inference

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-neural.txt
python code/neural_probe.py --refresh-cache
python code/label_control.py --refresh-cache
python code/review_audit.py
python code/report.py
```

Network access and original public model weights are required. Revisions are
pinned in `code/neural_probe.py`; weights are not redistributed. Cache reuse now
checks model revision and prompt SHA-256. Original archives lack prompt
fingerprints and are retained unchanged for explicit offline replay. The
inference runner rejects them unless refreshed. Do not retroactively attach
fingerprints and describe that as newly verified inference.

## Negative controls matter

* Oracle-nearest lowers Brier regret below the spread target, at higher KL and
  target mismatch. Target matching is not risk minimization.
* Audit-only-noisy32 and CAEA-noisy32 share the same target and report risk.
* Soft-score-8 improves risk and KL relative to half repair in this setting.
* Within-class preference preservation need not preserve aggregate style.
* Option permutations remove the pilots' initial positive DPO comparisons.
* An invariant wrong auditor remains wrong. Report classes can miss semantic
  cues. A finite grid cannot supply an unrestricted global curvature bound.

## Build and citation

`python code/build_paper.py` requires pdflatex and bibtex. The official-style
anonymous manuscript and proofs are in `submission/`; build main before appendix
because the latter reads its cross-reference file. Keep public author details,
citation files and this README out of the anonymous supplementary archive.

Original code is MIT licensed. That code license does not cover the manuscript
or model weights. TMLR style licensing is in `paper/THIRD_PARTY_NOTICE.md`; the
IJCAI 2027 style files are unchanged official files. The public paper reports
substantial generative-AI assistance in formalization, proofs, experiments,
analysis and writing; the named human author remains responsible for the work.
