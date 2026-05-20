# DeepSeek RLHF Records

Use `examples/deepseek_ranking_records.jsonl` as the append-only audit trail for
DeepSeek ranking outputs. Each line should be one prompt group with all sampled
candidates, the raw DeepSeek API response, and any parsed scores used for
offline ranking distillation.

Put your DeepSeek key in the local ignored `.env` file. If the file does not
exist in your container, create it from the committed template first:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
DEEPSEEK_API_KEY=your_real_key_here
```

Load it before running API calls:

```bash
set -a
source .env
set +a
```

Generate candidate groups from the current sampled CodePrior and optional
MetricRefiner:

```bash
python scripts/generate_deepseek_candidates.py \
  --stage1 stage1_rocstories_768_cosmos_best.pt \
  --vq vq_latent_tokenizer_rocstories_768_K1024_best.pt \
  --code_prior hier_code_prior_rocstories_768_K1024_plan8_best.pt \
  --metric_refiner hier_code_prior_rocstories_768_K1024_plan8_best_metric_refiner_best.pt \
  --num_prompts 64 \
  --candidates_per_prompt 8 \
  --sample_tau 0.9 \
  --rollout_steps 4 \
  --use_length_heads \
  --min_tokens 8 \
  --max_decode_tokens 48 \
  --output examples/deepseek_candidate_groups.jsonl
```

`--use_length_heads` requires a CodePrior checkpoint trained with the updated
`train_hier_code_prior.py`, which saves `valid_head` and `end_head` states. For
older checkpoints, omit that flag; candidates will decode all suffix slots.

Rank those groups with DeepSeek and append preference records:

```bash
set -a
source .env
set +a

python scripts/rank_deepseek_candidates.py \
  --input examples/deepseek_candidate_groups.jsonl \
  --output examples/deepseek_ranking_records.jsonl \
  --max_groups 10
```

Preferred recorder:

```bash
python scripts/record_deepseek_ranking.py \
  --prompt_id roc_val_0001 \
  --prompt "Prompt text here." \
  --candidate "candidate continuation 1" \
  --candidate "candidate continuation 2" \
  --deepseek_response deepseek_response_0001.json \
  --parsed_scores '{"ranking":["c1","c0"],"coherence":{"c0":2,"c1":4},"repetition":{"c0":1,"c1":5}}'
```

Record fields:

- `schema`: currently `deepseek_ranking_record_v1`
- `created_at_utc`, `run_id`, `prompt_id`
- `prompt`, optional `reference`, `source_file`
- `generator_checkpoint`, `refiner_checkpoint`
- `candidates`: candidate ids plus decoded text and optional metadata
- `judge`: DeepSeek provider/model and ranking prompt version
- `deepseek_raw_response`: exact API output text
- `deepseek_parsed_response`: JSON parse of the raw response when possible
- `parsed_scores`: cleaned ranking or scores used for training
- `notes`: manual caveats about the API call or candidate group

For first-pass distillation, keep the DeepSeek rubric stable: coherence, prompt
relevance, repetition, and ending quality. Do offline ranking/distillation from
this JSONL before attempting PPO.
