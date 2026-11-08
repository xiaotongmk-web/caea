# CAEA — When Pooled Evaluations Hide Belief Feedback

**Xiaotong Zhang**

[Public manuscript](paper/CAEA.pdf) | [Citation](CITATION.cff) | [Recorded data](results/) | [Changelog](CHANGELOG.md)

The current public revision adds an exact sampled-kernel result, a
two-sided finite-audit decision rule, and a reanalysis of archived model
outputs against stronger controls. The manuscript is an unpublished
research preprint.

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

## What changed in the current revision

- The binary sampled process has an exact one-step Wasserstein coefficient.
  Two stable deterministic equilibria can coexist with a unique stationary
  distribution for the sampled process.
- The finite-grid audit can certify contraction given a curvature bound,
  rule it out given a sufficiently steep measured secant, or return an
  inconclusive result.
- On the archived Bayesian task, the calibrated repair has forecast regret
  **0.08288**, versus **0.07142** for always reporting 0.5. The paired
  difference remains positive under 750 resamples that refit the calibrator.
  Full repair retains roughly **3%** of the analytic update magnitude.
  It has no clear update-error advantage over zero update.
- The paired signed stance contrast points in opposite directions for
  Qwen2.5-1.5B and Qwen2.5-7B. Report the signed and absolute contrasts
  separately. These are measurements on two audited stances, not global
  stability estimates.

## Recent reference methods and benchmarks

| Work | Relevant comparison |
| --- | --- |
| [How RLHF Amplifies Sycophancy](https://proceedings.mlr.press/v306/shapira26a.html) | A formal amplification mechanism and a minimum-KL correction. |
| [BASIL](https://arxiv.org/abs/2508.16846) | Bayesian diagnostics and BayesSFT/BayesDPO for evidence-based updating. |
| [Sycophancy Suppression Can Impair Rational Updating](https://aclanthology.org/2026.findings-emnlp.1208/) | Separates unsupported yielding from rational updating. |
| [SycoBench-600](https://aclanthology.org/2026.findings-acl.1759/) and [Big-Big-SycoBench](https://arxiv.org/abs/2610.15634) | Tests correction selectivity under pressure or valid and invalid arguments. |
| [BayesBench](https://arxiv.org/abs/2606.30850) | Tracks belief trajectories as evidence accumulates across turns. |

These works motivate comparisons on matched tasks. The recorded results
below use the repository's controlled tasks. They provide no head-to-head
ranking on these external benchmarks.

## Research tools

See the [CPU replay and evidence-grid tools](docs/CPU_REPRODUCIBILITY_AND_NEXT_STAGE.md) for an isolated replay command and a prompt design generator.

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
python code/evidence_replay.py
python code/review_audit.py
python code/verify_scale.py
python code/verify_external.py
```

The first checks study consistency. The second replays the 4,672 archived Q4
records and regenerates the paired stance and update results, calibration-refit
intervals, and the manuscript table. The third replays archived
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

## Recorded multi-model study

| Component | Actual completed scope |
| --- | --- |
| Backbones | Qwen2.5-1.5B / SmolLM2-1.7B / Qwen2.5-7B Instruct, official Q4_K_M |
| Parameters | 1.54B / 1.71B / 7.61B |
| Bayesian contexts | 256 training / 64 validation / 128 independent test |
| Primary conditions | Base, permuted options, paraphrased evidence, informative new evidence |
| Primary fresh requests | 1,152 common-auditor + 1,024 selector per backbone = 4,224 |
| External probes | 32 length-filtered GSM8K-derived questions, 16 correct / 16 wrong numeric claims |
| External fresh requests | 64 common-auditor + 128 selector per backbone = 448 |
| Independent test units | 128 Bayesian contexts plus 32 source questions; prompts/model repeats add no units |
| Comparisons | All fourteen methods and all protocol coefficients |
| Uncertainty | 5,000 paired context resamples; 500 historical auditor-refit resamples and 750 for the current reanalysis |

The locally fixed execution protocol precedes study inference but is not an
external preregistration. Later controls and the external extension are
explicitly exploratory. The frozen Qwen-1.5B common auditor sees evidence
without stance, in two complementary answer orders. Only two calibration
parameters are fit. Training labels fit them; validation labels select
regularization; test labels evaluate non-oracle policies only. The external
calibrator is transferred unchanged.

**The expanded results do not establish superiority over simple shrinkage.**
Full Bayesian calibrated-target regret is 0.0829 versus constant-target
0.0840; the refitted difference interval includes zero. External calibrated
target regret is 0.2785 versus constant 0.2626. Full repair nearly suppresses
informative updating (mean absolute update 0.0087 versus true 0.2931).
All full-repair backbones and audit-only share report risk by construction.
Option-order and low-candidate-mass failures remain included. This is a
controlled audit-quality failure study, not free-text model benchmarking.

Reanalyse the archived native scores without model downloads:

```bash
python code/scale_study.py analyse
python code/external_study.py analyse
python code/auditor_refit_bootstrap.py --replicates 500
python code/scale_report.py
python code/evidence_replay.py
python code/verify_scale.py
python code/verify_external.py
```

Fresh CPU inference uses the official wheel index, pinned weights and prompt
archives. The three downloads total roughly 6.9 GB; the 7B quantized model
requires RAM beyond its 4.7 GB weight size. Run one model process at a time
on a memory-limited machine.

```bash
python -m pip install --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements-core.txt -r requirements-scale.txt
python code/scale_models.py --directory /tmp/caea-weights
python code/scale_study.py infer --model qwen15 --weights /tmp/caea-weights --cache /tmp/caea-cache --threads 4
python code/scale_study.py infer --model smol17 --weights /tmp/caea-weights --cache /tmp/caea-cache --threads 4
python code/scale_study.py infer --model qwen7 --weights /tmp/caea-weights --cache /tmp/caea-cache --threads 4
python code/external_study.py infer --model qwen15 --weights /tmp/caea-weights --cache /tmp/caea-external-cache --threads 4 --shard-count 1
```

Repeat the external command for `smol17` and `qwen7`. Use a fresh cache when
execution metadata changes. The workflows show four disjoint context shards
per model; `merge --artifacts DIRECTORY --shard-count 4` checks complete
coverage, revision/engine consistency and fingerprints before combining.
The current llama binding's live native logits are used, rather than an
unpopulated `model.scores` array. Candidate mass and greedy-letter adherence
are archived. No post-hoc format exclusions are made.

The external task is transformed numeric-claim correctness on a small,
short-question subset, not standard GSM8K solve accuracy. Possible benchmark
pretraining exposure is not excluded. Checkpoints, family, training and
quantization confound causal effects of model scale or RLHF.

## Pending GPU replication

An exploratory [public-evidence diagnostic](results/auditor_diagnostics/)
now compares the learned auditor to a text-only Bayes-rule solver. On the
same 128 contexts the formula target attains base regret 0.01466 versus
0.08288 for the calibrated LM target, and preserves 98.37% rather than 2.96%
of the aggregate informative-update magnitude. It reads public prompt
evidence, not test labels. This task-specific competitor and the exact risk
decomposition further expose the audit bottleneck; they are post-hoc replay,
not new model inference or general language evidence.

```bash
python code/evidence_audit_diagnostics.py
python -m unittest discover -s code -p test_research_integrity.py -v
```

[gpu/RENTAL_PLAN.md](gpu/RENTAL_PLAN.md) proposes a single RTX 4090 and a
separate BF16 replication. `code/gpu_study.py` and `gpu/run_autodl.sh` are
prepared, with pinned original HF revisions and incremental raw-score logs.
No GPU has been rented or run by this artifact. The runner now defaults to
an eight-request smoke test, uses a separate smoke directory, checks CUDA/BF16
and free disk before downloading weights, and validates resumed raw-score
archives. Full analysis rejects incomplete and smoke archives. These paths
have CPU integrity tests only; they must pass actual CUDA validation before
their results are included. Rental requires the owner's account and spending
ceiling. Verify a provider-side shutdown timer before launching the script.
The staged [research plan](RESEARCH_PLAN_zh.md) defines the decision gates and
additional evidence required for a stronger submission.

## Original small frozen-model inference

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

Run `python code/build_paper.py` with pdflatex and BibTeX to build the
[public manuscript](paper/CAEA.pdf) from the source in [paper/](paper/).
Use [CITATION.cff](CITATION.cff) for citation metadata.

The original code is MIT licensed. The manuscript, third-party style files,
and model weights have separate terms. See
[the style notice](paper/THIRD_PARTY_NOTICE.md) for the bundled TMLR files.
The public paper documents generative-AI assistance and author responsibility.
