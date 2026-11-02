# Exploratory external numeric-claim correctness probes

Not standard GSM8K solve accuracy. Select the first 32 source-order test
questions with <=450 characters and integer final answers. Alternate one
correct or deterministically wrong numeric-answer claim per question.
The pinned source revision, whole-file hash, indices, prompts and selection
rule are in protocol.json. Selected published questions/solutions are in
source_selected.jsonl.gz under GSM8K_LICENSE.txt. The same Bayesian auditor
calibrator is transferred unchanged; external labels never fit or select it.
Base/permuted conditions and three models reuse the same 32 questions.
There are 448 unique native queries, not 448 independent questions.
