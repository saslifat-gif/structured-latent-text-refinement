# Structured Latent Text Refinement

**Structured Latent Text Refinement** is an experimental research project for non-autoregressive text refinement in continuous BERT latent space.

The project studies a central problem in latent text generation:

> Continuous latent similarity is not enough for text.  
> A generated latent can be geometrically close to a real latent but still decode into wrong or collapsed tokens.  
> Text generation requires preserving discrete token identity inside the latent source.

This repository contains the current prototype pipeline:

```text
structured draft
→ BERT latent space
→ DraftPrior
→ local Riemannian / structured flow refinement
→ parallel decoder
→ text
```

This is **not** a general chatbot or GPT replacement.  
The current working mode is **parallel latent refinement from a structured draft**.


---

## Current Status

The project has reached the first readable-refinement milestone.

Earlier versions failed from pure Gaussian or prompt-only starts:

```text
pure Gaussian / weak prompt prior → punctuation, repeated words, generic fragments
```

The current system works when given a structured draft:

```text
rough draft tokens
→ local Gaussian latent scaffold
→ DraftPrior
→ small residual Riemannian flow
→ readable but imperfect text
```

The model can now produce sentence-like refinements, but still makes factual and lexical substitutions. The main remaining problem is **exact token identity preservation**, not basic language emergence.

---

## Core Idea

Image latent generation can often start from broad Gaussian noise. Text is different.

In this project, experiments suggest:

```text
100% draft structure → strong decoder readability
95% draft structure  → useful but unstable
90% draft structure  → weak but trainable
pure noise           → collapse / garbage
```

This motivates a local source distribution:

```text
z_start = alpha * z_draft + noise
```

instead of a global Gaussian start.

The system therefore treats language generation as:

```text
discrete scaffold
→ local continuous latent cloud
→ parallel latent refinement
```

rather than:

```text
global Gaussian noise
→ text from scratch
```

---

## Pipeline

### Stage 1: BERT latent autoencoder

Stage 1 encodes text with a frozen BERT encoder and trains a parallel decoder to reconstruct text from compressed latent sequences.

```text
tokens
→ frozen BERT encoder
→ latent sequence z
→ parallel decoder
→ reconstructed tokens
```

The decoder creates the latent text manifold used by later stages.

### DraftPrior

The DraftPrior learns to map a draft-like latent into a decoder-readable latent.

```text
z_draft + local noise
→ DraftPrior
→ refined latent start
```

The key training curriculum uses gradually corrupted drafts:

```text
100% real draft
→ 95% real draft
→ 90% real draft
```

This teaches the prior to preserve token identity before handling harder corruption.

### Stage 2: Local Riemannian refinement

Stage 2 uses a prompt-conditioned FlowNet and MetricNet to perform small residual latent refinement.

Earlier full-transport flow damaged good DraftPrior starts. The current formulation uses local refinement:

```text
v_target = small_fraction * (z_real - z_start)
```

and a small ODE update scale:

```text
z = z + FLOW_REFINE_SCALE * v * dt
```

This reframes the flow as a local editor, not a full generator.

---

## Repository Layout

```text
configs/
  YAML snapshots of important experiment settings.

src/
  Core model, loss, evaluation, and inference modules.

scripts/
  Training and inference entrypoints.

examples/
  Sample outputs, failure cases, ablation notes.

docs/
  Project changelog and standalone setup notes.

paper/
  Draft paper notes and experimental writeups.
```

---

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

The project expects PyTorch, Transformers, and common ML utilities.

Run commands from the repository root so checkpoints are saved and loaded from the expected paths.

---

## Quick Start

Train Stage 1:

```bash
python scripts/train_stage1.py
```

Train the DraftPrior:

```bash
python scripts/train_draft_prior.py
```

Train Stage 2 refinement:

```bash
python scripts/train_stage2.py
```

Run inference:

```bash
python scripts/run_inference.py \
  --stage1 stage1_best.pt \
  --stage2 stage2_conditional_flow_decoder_joint_best.pt \
  --chatbot
```

---

## Checkpoints

Checkpoints are ignored by Git.

Expected local checkpoint names:

```text
stage1_best.pt
draft_prior_best.pt
stage2_conditional_flow_decoder_joint_best.pt
```

Place them in the repository root before running inference.

Large `.pt` files should not be committed to Git. Use a model host such as Hugging Face Hub for checkpoint distribution.

---

## Example Usage

The current model works best with a prompt and a rough draft.

Example:

```text
prompt:
quantum mechanics describes

rough draft:
particles act like waves and measurements are described by probabilities
```

The model performs parallel latent refinement over the draft and returns:

```text
prior output
flow output
fused output
```

The blank prompt-only path is not the main working mode. A structured draft is required for reliable behavior.

---

## What Works

The system currently demonstrates:

```text
structured draft → readable latent refinement
```

Positive findings:

- DraftPrior can preserve token identity when the draft is highly structured.
- Pure Gaussian starts fail for text latent generation.
- Local Gaussian starts around draft latents are much more decoder-readable.
- Riemannian flow helps only when trained as a small residual refiner.
- Large ODE transport destroys token identity.
- Parallel decoding becomes viable once the latent remains inside the correct token basin.

---

## Current Limitations

The model is still experimental.

Known limitations:

- It is not a prompt-only chatbot.
- It requires a structured rough draft.
- It can substitute wrong words while preserving sentence shape.
- It may preserve grammar but alter facts.
- Fused output is not always better than flow output.
- Draft generation is currently the missing component for full chatbot-style use.
- The best current framing is text refinement, not open-ended generation.

---

## Research Framing

The main research claim is:

> Text latent generation requires preserving discrete token identity.  
> Continuous latent geometry alone is insufficient.  
> Structured local latent scaffolds are more effective than global Gaussian starts.

A concise framing:

```text
Parallel latent text refinement from structured drafts.
```

Not:

```text
Prompt-only parallel text generation from scratch.
```

---

## Suggested Paper Title

```text
Preserving Discrete Token Identity in Continuous Latent Text Generation
```

Alternative:

```text
Structured Local Latent Refinement for Non-Autoregressive Text Generation
```

---

## Suggested Experiments

Important comparisons:

```text
Pure Gaussian start
Prompt-only prior
Structured alpha start
DraftPrior only
DraftPrior + local flow
DraftPrior + flow + fusion
```

Important metrics:

```text
target token probability
top-1 accuracy
decoder cross entropy
collapse ratio
unique token ratio
sample readability
```

Important ablation:

```text
100% draft
95% draft
90% draft
pure noise
```

---

## Development Notes

Runtime constants currently live partly inside Python modules, especially:

```text
src/stage2_config.py
```

The YAML files in `configs/` are experiment snapshots for readability and future config loading.

The project is still research code. Interfaces and configs may change.

---

## License

```text
Apache-2.0
```

---

## Citation

No formal citation yet now.
