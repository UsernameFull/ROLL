# RLVR First-Update Logprob Diagnostics

## Goal

Add a narrowly scoped, read-only diagnostic that exposes train/infer logprob
alignment and PPO movement before and after the first optimizer update. The
diagnostic must not change loss computation, masks, optimizer state, or existing
metric aggregation.

## Scope

- Run only when `global_step == 0`.
- Emit detailed logs only from actor-training rank 0.
- Report each optimizer batch separately so the existing average across eight
  optimizer batches does not hide the first-update behavior.
- Limit token-level detail to the first two local samples and first 32 valid
  response tokens.
- Use the existing worker logger; do not add configuration or output files.

## Data and Metrics

For every optimizer batch at pipeline step zero, log:

- optimizer `batch_idx`;
- ratio mean, median, p01, and p99;
- approximate KL and signed policy KL;
- low-side, high-side, and total PPO clip fractions;
- current learning rate when available.

For the first optimizer batch only, log bounded token records containing:

- sample index and response-token position;
- token ID;
- current actor logprob;
- recomputed old actor logprob;
- vLLM inference logprob;
- BF16 reference logprob;
- current/old PPO ratio and old/inference train-infer ratio.

All token selections use the response mask. Tensor values are detached before
conversion to host values.

## Implementation

`roll/pipeline/rlvr/actor_worker.py` will compute and emit the diagnostic inside
the existing loss function, where current, old, inference, and reference
logprobs are simultaneously available. A transient optimizer-batch index will be
placed in `DataProto.meta_info` by `roll/pipeline/base_worker.py` immediately
before each strategy training call. This avoids changing the loss function
interface or public protocol.

The log prefix will be stable (`[RLVR_LOGPROB_DIAG]`) so records can be extracted
with `rg`. Failures in diagnostic formatting must never interrupt training; the
diagnostic will catch formatting/indexing errors and log one warning.

## Verification

- Unit-test the pure statistics/record-building helper with synthetic tensors.
- Verify response-mask filtering and the two-sample/32-token limits.
- Verify the diagnostic activation predicate accepts only global step zero,
  optimizer batches on rank zero.
- Run Python syntax compilation on changed modules.

## Non-goals

- Full-vocabulary logits or exact full-distribution KL.
- Full tensor dumps or JSONL artifacts.
- Changes to sampling parameters, train-infer correction, PPO, or FP8 kernels.
- Enabling the diagnostic beyond the first pipeline step.
