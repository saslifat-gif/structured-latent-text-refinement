# Failure Cases

This file summarizes recurring failure modes from `docs/CODEx_CHANGELOG.md`.
It is intended as a checklist for reading samples, logs, and benchmark outputs.

## Token Collapse And Repetition

Symptom:

- Generated suffixes repeat common words or fall into a dominant-token basin.
- Argmax decoding can look much worse than sampled decoding.
- Validation cosine or CE may improve while text quality remains repetitive.

Changelog evidence:

- Token-collapse diagnostics were added after samples stayed repetitive despite
  improving cosine and raw CE.
- Higher decoder CE, oracle-logit KL, entropy shaping, and generated-logit
  balance each helped diagnose the problem but could also create new token
  attractors.

What to inspect:

- Unique-token ratio.
- Maximum token fraction.
- Dominant generated tokens.
- Generated vs oracle entropy.
- Target-token probability and top-1 accuracy.

## Decoder-Manifold Drift

Symptom:

- A generated latent is geometrically close to the real latent but decodes to
  wrong or collapsed tokens.
- Decoder CE remains poor even when latent cosine improves.

Changelog evidence:

- Hidden-state matching, oracle-logit KL, target-probability losses, and
  projector diagnostics were all attempts to measure or repair this gap.
- Interpolation diagnostics were added to test whether small moves toward the
  real latent restore decoder compatibility.

What to inspect:

- Raw generated CE vs oracle real-latent CE.
- Small-alpha interpolation CE.
- Target-token probability under generated logits.
- Whether fixes improve text or only latent geometry.

## Bad Start Distribution

Symptom:

- Pure Gaussian or blank latent-chain starts produce punctuation, repeated words,
  generic fragments, or unrelated Wiki-like text.
- Prompt-only starts may not contain enough continuation signal.

Changelog evidence:

- Blank latent fallback was disabled as the default inference demo.
- The system was reframed around structured drafts, DraftPrior starts, and local
  continuous refinement.
- A May 18 prompt-only prior diagnostic was added to test whether prompt latents
  beat Gaussian suffix starts before adding more Stage 2 complexity.

What to inspect:

- Draft source: real, synthetic, manual, retrieval, template, prompt-only, or
  Gaussian.
- Draft corruption rate and checkpoint metadata.
- DraftPrior-only metrics before ODE refinement.

## Flow Damages A Good DraftPrior Start

Symptom:

- DraftPrior output is readable or has good target probability, but full ODE
  refinement makes CE and samples worse.
- Velocity norms grow while rollout flow-token CE collapses.

Changelog evidence:

- Stage 2 was reframed from full latent transport into local residual
  refinement.
- `FLOW_REFINE_SCALE`, `FLOW_REFINE_TARGET_FRACTION`, velocity clamp, and metric
  freezing were added after large velocity norms damaged good starts.
- A refine-scale compensation bug once caused velocity inflation by training the
  scaled update to match the full residual.

What to inspect:

- DraftPrior CE / target probability vs post-flow CE / target probability.
- Velocity output norm.
- Metric min/max and regularization.
- Whether `FLOW_REFINE_SCALE` and target fraction match checkpoint metadata.

## Metric Misbehavior

Symptom:

- MetricNet either expands anisotropy too quickly or stays near identity.
- Metric changes correlate weakly with token improvement.

Changelog evidence:

- Metric log bounds moved from hard clamp to smooth `tanh`.
- Identity initialization, metric warmup, metric frozen steps, metric LR, and
  metric regularization became tunable.
- Option B corrected the train/inference semantics so the metric is meaningful
  for natural velocity rather than only a weighted loss.

What to inspect:

- Metric min/max and log bound.
- Metric regularization multiplier.
- Whether checkpoint predates the Option B natural-velocity fix.

## Evaluation Or Packing Mismatch

Symptom:

- Benchmark results look unexpectedly bad or unavailable.
- Prompt/suffix slots are misaligned, especially in 64/64 ROCStories runs.
- Checkpoint loading fails due to shape or dtype mismatch.

Changelog evidence:

- ROCStories benchmark packing was fixed so prompts occupy prompt slots and
  references/drafts occupy suffix slots.
- Inference now reads prompt length and max sequence length from Stage 2
  checkpoints.
- Batched benchmark dtype mismatch was fixed by using autocast for DraftPrior.

What to inspect:

- `PROMPT_LEN`, `MAX_SEQ_LEN`, latent dimension, and checkpoint metadata.
- Whether benchmark uses fixed slot packing rather than `prompt + draft`
  tokenized as one ordinary sequence.
- Whether unavailable rows contain flattened exception text in the CSV.

## Reporting Ambiguity

Symptom:

- It is unclear which Stage1, DraftPrior, or Stage2 checkpoint produced a sample.
- Different runs appear comparable but used different latent width, dataset,
  draft corruption, or prompt/suffix split.

Changelog evidence:

- Inference now prints resolved checkpoint paths, metadata, and fingerprints.
- DraftPrior checkpoints are corruption-aware.
- Benchmark CSV writing and export files were hardened for reproducibility.

What to inspect:

- Stage1 path.
- Stage2 path.
- DraftPrior path and corruption tag.
- Dataset, split strategy, prompt slots, target slots, and latent dimension.
