# Expanded study execution protocol

The fixed protocol and case/request fingerprints were written before fresh study inference.
This is a locally fixed execution plan, not an externally registered preregistration.

- 256 calibration-training, 64 validation and 128 independent test evidence contexts.
- 3 frozen, Q4_K_M model checkpoints: Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, Qwen2.5-7B-Instruct.
- 1,152 evidence-only auditor queries and 1,024 selection queries per backbone: 4,224 planned fresh queries.
- Same 128 test contexts in original order, permuted answers, paraphrased evidence and informative additional evidence conditions.
- Two unsupported stances share evidence, wording preference and option permutation within each context.
- The common auditor uses Qwen1.5B evidence-only responses averaged over two answer orders.
- A two-parameter calibrator is fit on training contexts; regularization is selected using validation risk only.
- Methods never receive test posterior labels; those are used only for evaluation and explicitly labeled oracle controls.
- Every repair fraction and soft-score coefficient in protocol.json is reported without test-based winner selection.
- Confidence intervals resample evidence contexts, not the two dependent stance prompts. They hold the calibration split and checkpoints fixed.
- Candidate probabilities are normalized over supplied answer letters. Full-vocabulary candidate mass is also recorded. No free-text generation claim is made.
- The models are quantized; parameter scaling is not isolated from training differences. No model weights are trained here.

If resource constraints require an amendment, it must be separately recorded. Missing runs cannot be represented as completed experiments.
