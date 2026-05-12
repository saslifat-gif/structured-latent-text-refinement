# Prompt-Conditioned Riemannian Flow Matching in Frozen Language Latent Space

## Abstract

This draft studies text continuation by refining suffix latents in the frozen
latent space of a BERT-based encoder-decoder. The central hypothesis is that a
learned diagonal Riemannian metric can weight latent directions according to
their downstream textual significance, improving flow-matching refinement over a
flat Euclidean objective.

## Method Sketch

Stage 1 learns a latent autoencoder for token sequences. Stage 2 conditions on
prompt latents and learns a velocity field from noisy suffix latents toward clean
suffix latents. A metric network predicts a positive diagonal metric over the
same latent coordinates and weights the flow-matching residual.

## Open Questions

- How much of the improvement comes from the metric versus the draft prior?
- Does the learned metric correlate with decoder sensitivity?
- Which failure modes remain after auxiliary token and rollout losses?
