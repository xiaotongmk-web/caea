# Public revision 0.4.0 — 2026-11-08

- Updated the public manuscript and citation metadata.
- Added an exact sampled-kernel contraction coefficient and the distinction between
  deterministic equilibria and stationary distributions.
- Added a finite-grid decision simulation that can certify, refute, or abstain.
- Added signed stance contrasts, the deterministic 0.5 baseline, and 750
  calibration-refit resamples. Reused the original archive for the reanalysis.
- Added recent reference benchmarks and clarified the scope of comparability.

# Unreleased continuation - 2026-11-02

- Added a public-evidence Bayes solver as a task-specific, label-free
  competitor, plus exploratory archived-score risk and update diagnostics.
- Documented that the calibrated auditor retains only 2.96% of aggregate
  informative-update magnitude; the formula target retains 98.37%.
- Added strict GPU-cache validation and tests for duplicate/corrupt scores,
  prompt mismatches, incomplete archives and partial last writes.
- Added CUDA/BF16/disk preflight and a separate eight-request smoke stage.
- Added staged scientific decision gates; no GPU has yet been rented or run.

# Experimental expansion 0.3.0 - 2026-11-02

- Executed 4,224 unique fresh Bayesian requests on frozen official 1.54B/1.71B/7.61B Q4_K_M checkpoints, plus 448 exploratory external requests.
- Split 256/64/128 independent Bayesian calibration-training/validation/test contexts; fit only two auditor calibration parameters, without test labels.
- Added 32 GSM8K-derived numeric-claim correctness questions, with unchanged transferred calibration and no external-label fitting.
- Added constant-target, pooled-stance, raw-auditor and nearest-report controls; report all four fractions and three soft-score coefficients.
- Added 5,000 paired context resamples and 500 train/validation/test auditor-refit resamples, including validation regularization reselection.
- Archived native logits, candidate mass, prompt/protocol fingerprints, pinned upstream SHA-256 files and completed shard metadata.
- Retained weak auditor performance, attenuated informative updates, option-order failures and the original small head-DPO pilot.
- Revised public and anonymous manuscripts around observability, target quality, exact repair identities and failure analysis.
- Added a pending BF16 CUDA replication runner and costed GPU rental proposal; no GPU run or rental is represented as completed.
- No full-model post-training, standard GSM8K solve accuracy, causal scaling, free-text or human-benefit claim is made.

# Review revision 0.2.0 - 2026-11-02

- Added an explicit low-regret audit-blindness extension and proof.
- Added linear risk / target-TV identities and clarified the well-specified synthetic learner.
- Tightened the finite-grid curvature remainder to H*h/2 and regenerated its experiment.
- Added matched noisy-audit and nearest-report comparators; regenerated all synthetic runs.
- Replayed archived neural predictions with 24-context / 5-seed crossed bootstrap intervals.
- Rejected infeasible empty-class repair and bound future inference cache reuse to prompts and revisions.
- Added offline audit, small demo, reproducible public-paper builder and expanded CI checks.
- Prepared a 7-content-page anonymous IJCAI 2027 manuscript with separate proofs using official style.
- Kept option-permutation failures, oracle dependence and human/free-text limitations explicit.
- Neural backbone inference was not rerun in this revision; archived arrays are retained unchanged.

# Changelog

## 0.1.0 - 2026-11-01

Initial public-artifact preparation: controlled DPO experiments, stochastic feedback simulations, finite-audit controls, frozen-neural candidate-selection pilots and option permutations. No publication or acceptance is asserted.
