# Structured Latent Text Refinement

Repository: https://github.com/saslifat-gif/structured-latent-text-refinement

This is the standalone home for the BERT latent-refinement project. It packages
the current training, inference, configs, examples, and paper notes outside the
older `ml-foundations` workspace.

The current milestone is a two-stage system:

1. Stage 1 trains a frozen-BERT latent autoencoder.
2. A draft prior predicts a useful suffix latent draft from the prompt.
3. Stage 2 learns a prompt-conditioned structured/Riemannian flow that refines
   suffix latents before decoding them back to text.

## Layout

```text
configs/   YAML snapshots of the main experiment settings.
src/       Model, loss, evaluation, and inference modules.
scripts/   Entrypoints for training and inference.
examples/  Readable notes for samples, failures, and ablations.
docs/      Project changelog.
paper/     Draft paper notes.
```

See `docs/STANDALONE.md` for the standalone Git setup.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run commands from this directory so checkpoints are written here:

```bash
python scripts/train_stage1.py
python scripts/train_draft_prior.py
python scripts/train_stage2.py
python scripts/run_inference.py --stage1 stage1_best.pt --stage2 stage2_conditional_flow_decoder_joint_best.pt --chatbot
```

The Python training scripts currently own their runtime constants in
`src/stage2_config.py` and module-level settings. The YAML files are checked-in
experiment snapshots for readability and future config loading.

## Checkpoints

Model checkpoints are intentionally ignored by Git. Expected local names:

```text
stage1_best.pt
draft_prior_best.pt
stage2_conditional_flow_decoder_joint_best.pt
```

Keep them in the project root when running the scripts above.
