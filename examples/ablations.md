# Ablations

This file summarizes the main ablation threads recorded in
`docs/CODEx_CHANGELOG.md`. It is a compact map of what was tried and why the
current pipeline settled on structured-draft latent refinement.

## Start Distribution

The largest practical ablation was the source distribution for Stage 2.

- Pure Gaussian or weak prompt-only latent starts repeatedly decoded into
  punctuation, repetition, generic fragments, or unrelated Wiki-like text.
- Structured draft starts were much more readable.
- Corrupted draft starts showed a usable ladder: clean drafts were easiest, 5%
  dropout was useful, and 10% dropout was harder but trainable.
- Blank latent-chain fallback was demoted to a diagnostic path because it did not
  present the system honestly.

Takeaway: the working system needs a local draft-like latent scaffold. Stage 2 is
best treated as refinement, not free-form latent generation from global noise.

## Riemannian Flow Formulation

Several flow objectives were explored before the current Option B interpretation.

- Early metric-weighted losses improved some latent scores but left a semantic
  mismatch: training acted like Euclidean velocity prediction while inference
  used metric division.
- The current protected setup makes `FlowNet` predict a covector-like field and
  uses natural velocity `v = f / g` during rollout and inference.
- Older checkpoints trained under weighted-velocity semantics should be treated
  as pre-fix ablations.

Takeaway: metric division is intentional. Flow supervision should match the
natural-velocity sampler.

## Rollout Length And Locality

The rollout path moved from full transport toward local residual editing.

- Increasing rollout training from 4 to 8 steps improved inference matching in
  principle, but an early 8-step / smaller-batch run regressed validation score,
  raw CE, and samples.
- Later Stage 2 runs reframed the flow as a small local refiner after the
  DraftPrior start, with explicit `FLOW_REFINE_SCALE`,
  `FLOW_REFINE_TARGET_FRACTION`, velocity clamping, and metric controls.
- A compensation bug once made the model learn velocities inflated by
  `1 / FLOW_REFINE_SCALE`; fixing it made the objective genuinely local.

Takeaway: full latent transport damaged otherwise useful starts. Small,
controlled residual flow is the healthier Stage 2 role.

## Decoder-Side Pressure

Several decoder losses and adaptation variants were tested to address readable
latent mismatch.

- Larger decoder CE weights made token pressure stronger but could inflate total
  loss, regress cosine, and trigger repetition.
- Hidden-state matching and oracle-logit KL probes helped diagnose decoder
  mismatch but did not reliably fix token collapse.
- Entropy shaping sometimes improved distribution sharpness briefly, but could
  oversharpen into wrong-token attractors.
- Controlled decoder adaptation was kept as a labeled experiment, but the main
  architecture still relies on frozen Stage1 plus Stage2 refinement.

Takeaway: decoder pressure is useful as a judge and diagnostic, but too much
token-side pressure can create brittle attractors.

## Diversity, OT, And Geometry Regularizers

Geometry regularizers were added to fight collapse and manifold drift.

- Norm preservation helped prevent latent norm collapse.
- Pairwise token-level diversity was more useful than sequence-mean diversity
  for exposing token-basin collapse.
- Optional OT regularization was added for 768-dimensional ROCStories runs as a
  distribution-level latent regularizer.
- Metric regularization, metric warmup, and metric LR controls were exposed after
  MetricNet either expanded too quickly or stayed too close to identity.

Takeaway: geometry guardrails help, but they do not replace a good local start
distribution.

## Capacity And Dataset Ablations

The experiments moved from WikiText-style fixed-token splits toward ROCStories
sentence continuation.

- ROCStories retraining uses first two sentences as prompt and final three as
  target continuation.
- Stage1, DraftPrior, Stage2, inference, and benchmark code now support
  configurable latent dimensions.
- The 768-dimensional path became the default ROCStories setting so it can be
  tested as a true high-capacity run rather than a placeholder benchmark row.

Takeaway: data split and latent width matter enough that comparisons should
state dataset, prompt slots, suffix slots, and latent dimension.

## Current Recommended Baseline

For current Stage 2 comparisons, use:

- Stage1 frozen BERT latent autoencoder.
- ROCStories sentence split when evaluating story continuation.
- DraftPrior start with controlled corruption metadata.
- Option B natural-velocity `FlowNet + MetricNet`.
- Local residual refinement with conservative step size.
- Separate reporting for autoregressive baselines, reported Diffusion-LM rows,
  local Diffusion-LM generations, and synthetic-draft-conditioned latent
  refinement.
