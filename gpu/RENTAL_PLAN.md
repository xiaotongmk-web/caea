# GPU rental proposal - 2026-11-02

Status: prepared only. No instance has been rented or charged by this package.
The completed paper uses actual CPU Q4_K_M results. GPU numbers must not be
added to it until the pending runs finish and their archives pass review.

Recommended first stage: AutoDL, one RTX 4090 24 GB, on-demand, 32 GB or more
host RAM, at least 50 GB usable local disk, PyTorch-compatible Linux image.
Official public list price checked today: RMB 1.88/hour, subject to available
host/checkout price. Select a host at or below RMB 2.50/hour. A four-hour
instance timer bounds compute to RMB 10 at that ceiling, assuming the timer
works. Downloads, setup, failures and idle time count as running time. Paid
expanded disk is separate, and may keep charging after compute shutdown.
A proposed RMB 50 overall budget is a reserve, not a guaranteed cost or an
authorization already obtained. Initial account top-up may have its own
minimum. No automatic renewal or prepaid multi-day package is needed.

Runpod Secure RTX 4090 is an alternative at the displayed USD 0.89/hour;
account access, checkout availability and residual storage charges must be
checked before rental. Do not assume either provider has inventory.

Purpose: non-quantized BF16 CUDA replication on the same 1.5B/1.7B/7B model
families and exact question/stance protocol. This tests precision/runtime
robustness. It adds no independent questions and cannot by itself establish
free-text or human benefits. HF and GGUF runtime/template/base-revision
differences can contribute, so it is not an isolated causal quantization test.
The pinned public HF model revisions are in code/gpu_study.py.

Before creating an instance, the owner must supply the spending ceiling and
sign in to their chosen provider. Use the provider's authentication flow;
do not put account passwords, payment-card data or API keys in the paper,
GitHub, logs or research artifacts. The current connected-tool catalog does
not expose a GPU rental provider. No credentials are included in this repo.

Set a provider-side timed shutdown for four hours BEFORE starting work.
Download/setup are included in that period. Do not rely on script timeout
alone: a stopped Python process does not necessarily stop instance billing.

Clone the repository onto the instance and run:

```bash
# Optional AutoDL-only shutdown after either completion or failure:
export CAEA_AUTODL_SHUTDOWN=1
bash gpu/run_autodl.sh
```

The runner saves native logits incrementally and packs partial/complete
results before shutdown. The archive name does not assert completion;
check each model JSON and request count. Retrieve results from the provider,
verify them locally, and confirm instance status is stopped. GPU inference
has not been tested in this CPU-only workspace; syntax, CLI preparation and
the data protocol can be checked here. Do not describe that as a GPU run.

Sources:
- https://www.autodl.com/
- https://www.autodl.com/docs/price/
- https://www.autodl.com/docs/quick_start/
- https://www.autodl.com/docs/save_money/
- https://www.runpod.io/pricing
