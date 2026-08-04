# vLLM Teacher-Forced Inference Diagnostics

## Goal

Locate train/inference log-probability differences without comparing sampled text. For a bounded number of first-step rollout requests, score the exact generated token sequence twice inside vLLM:

1. normal incremental decode, using the log probabilities already returned by generation;
2. teacher-forced prompt scoring, by submitting `prompt_ids + output_ids` with `prompt_logprobs`.

Carry both aligned tensors into the actor batch and emit a driver-side `[RLVR_INFER_DIAG]` record comparing vLLM decode, vLLM teacher-forced scoring, and Megatron's recomputed old-policy log probabilities.

## Scope

- Text-only vLLM rollout requests.
- The first completed request per vLLM strategy instance by default.
- The first completion within that request; remaining completions carry `NaN` diagnostic values.
- Existing rollout behavior and sampled-token log probabilities remain unchanged.
- Multimodal requests are skipped because replaying processor inputs requires a separate design.

## Configuration

The vLLM `strategy_config` accepts:

- `rlvr_infer_diagnostic_requests`: number of requests to teacher-force per strategy instance; default `1`, set `0` to disable.

The option is removed before constructing vLLM and therefore is not passed to `AsyncLLM`.

On vLLM versions that support cache salts, the scoring request uses a unique salt so it cannot reuse the generation request's cached prefix. On older versions, users should set `enable_prefix_caching: false` to guarantee a cold prefill.

## Data Flow

1. `VllmStrategy.generate_request` performs normal generation and extracts sampled-token log probabilities.
2. An async claim guard bounds diagnostic requests under concurrent scheduling.
3. The claimed text request submits the concatenated prompt and first completion with `temperature=0`, `max_tokens=1`, and `prompt_logprobs=1`.
4. The selected-token prompt log probabilities are sliced to the completion span. Failures produce `NaN` values and a warning without failing rollout.
5. `RouterClient` preserves `teacher_forced_output_logprobs` in response metadata.
6. rollout post-processing aligns the values with `infer_logprobs` and stores `infer_teacher_logprobs` in the batch.
7. the actor's first optimizer batch builds driver diagnostics over finite teacher-forced positions.

## Driver Record

`[RLVR_INFER_DIAG]` contains:

- rank, global step, and optimizer batch index;
- finite diagnostic token count;
- vLLM decode minus teacher-forced delta statistics;
- Megatron old-policy minus teacher-forced delta statistics;
- Megatron old-policy minus vLLM decode delta statistics;
- bounded per-token records with sequence position and token ID.

Each delta summary reports mean, RMS, absolute p50/p95/p99/max, nonfinite fraction, and the fraction whose probability ratio is outside `[0.8, 1.2]`.

## Failure Handling

- Missing prompt-logprob entries, unsupported vLLM results, scoring exceptions, and unscored completions are represented with `NaN`.
- Diagnostic failures log a warning and never change generation outputs or terminate training.
- If no finite teacher-forced values reach the actor, no driver diagnostic is emitted.

## Tests

- Extract chosen-token values from vLLM prompt-logprob structures and preserve alignment.
- Bound concurrent diagnostic claims.
- Map teacher-forced completion log probabilities into the padded `[B, T-1]` batch layout.
- Build masked three-way delta summaries and bounded token records while ignoring `NaN` values.
- Preserve paused/continued rollout diagnostic metadata.

## Acceptance Criteria

- Existing generation results and `infer_logprobs` are unchanged.
- At most the configured number of requests per strategy instance perform an extra scoring request.
- `[RLVR_INFER_DIAG]` appears in driver logs when at least one scored response reaches the first optimizer batch.
- Targeted unit tests, Python compilation, lint checks, and whitespace checks pass.
