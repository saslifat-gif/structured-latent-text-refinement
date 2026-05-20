import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2_config import (
    BASE_NOISE_STD,
    CALIBRATE_GENERATED_LATENTS,
    CROSS_GATE_SCALE,
    FLOW_DEPTH,
    FLOW_HIDDEN_DIM,
    FLOW_REFINE_SCALE,
    GATE_INIT,
    GATE_REG_WEIGHT,
    MAX_SEQ_LEN,
    METRIC_HIDDEN_DIM,
    METRIC_LOG_BOUND,
    ODE_STEPS,
    PROMPT_LEN,
    SELF_GATE_SCALE,
    START_MLP_HIDDEN_DIM,
    START_NOISE_STD_FRAC,
    START_TRANSFORMER_HEADS,
    START_TRANSFORMER_HIDDEN_DIM,
    START_TRANSFORMER_LAYERS,
    STRUCTURED_START_ALPHA,
    STRUCTURED_TARGET_START,
    TARGET_LATENT_MEAN,
    TARGET_LATENT_STD,
    TOKEN_RESIDUAL_SCALE,
    TOKEN_SHARED_BLOCK,
    TOKEN_SHARED_BLOCK_SCALE,
    VELOCITY_CLAMP,
)


def clamp_velocity(v):
    if VELOCITY_CLAMP is None or VELOCITY_CLAMP <= 0:
        return v
    return VELOCITY_CLAMP * torch.tanh(v / VELOCITY_CLAMP)


class FlowNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=FLOW_HIDDEN_DIM, depth=FLOW_DEPTH):
        super().__init__()
        self.prompt_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prompt_pos = nn.Parameter(torch.zeros(1, PROMPT_LEN, hidden_dim))
        self.cond_proj = nn.Linear(PROMPT_LEN * hidden_dim, latent_dim)
        self.in_proj = nn.Linear(latent_dim * 2 + 2, hidden_dim)
        self.pos_proj = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "conv": nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=5,
                    padding=2,
                    groups=hidden_dim,
                ),
                "mix": nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                ),
                "self_norm": nn.LayerNorm(hidden_dim),
                "self_attn": nn.MultiheadAttention(
                    hidden_dim,
                    num_heads=8,
                    batch_first=True,
                ),
                "cross_norm": nn.LayerNorm(hidden_dim),
                "cross_attn": nn.MultiheadAttention(
                    hidden_dim,
                    num_heads=8,
                    batch_first=True,
                ),
            })
            for _ in range(depth)
        ])
        self.self_gates = nn.ParameterList([nn.Parameter(torch.tensor(GATE_INIT)) for _ in range(depth)])
        self.cross_gates = nn.ParameterList([nn.Parameter(torch.tensor(GATE_INIT)) for _ in range(depth)])
        self.token_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        self.token_residual_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.token_block[-1].weight)
        nn.init.zeros_(self.token_block[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.token_residual_proj.weight)
        nn.init.zeros_(self.token_residual_proj.bias)
        self._last_token_block_norm = torch.tensor(0.0)
        self._last_token_block_ratio = torch.tensor(0.0)
        self._last_token_hidden_norm = torch.tensor(0.0)
        self._last_velocity_out_norm = torch.tensor(0.0)
        self._last_out_proj_weight_norm = torch.tensor(0.0)

    def forward(self, z_t, t, z_cond, pos, mask=None, return_hidden=False):
        squeeze = z_t.dim() == 2
        if squeeze:
            z_t = z_t.unsqueeze(1)
            t = t.unsqueeze(1)
            if z_cond.dim() == 2:
                z_cond = z_cond.unsqueeze(1)
            pos = pos.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        prompt_h = None
        prompt_key_padding_mask = None
        if z_cond.dim() == 3 and z_cond.size(1) == PROMPT_LEN:
            prompt_h = self.prompt_proj(z_cond) + self.prompt_pos
            cond = self.cond_proj(prompt_h.reshape(prompt_h.size(0), -1))
            prompt_key_padding_mask = z_cond.abs().sum(dim=-1) == 0
            if prompt_key_padding_mask.all(dim=1).any():
                prompt_key_padding_mask = prompt_key_padding_mask.clone()
                prompt_key_padding_mask[prompt_key_padding_mask.all(dim=1), 0] = False
        elif z_cond.dim() == 3:
            cond = z_cond.mean(dim=1)
        else:
            cond = z_cond
        cond = cond.unsqueeze(1).expand(-1, z_t.size(1), -1)

        inp = torch.cat([z_t, cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        h = self.in_proj(inp) + self.pos_proj(pos.unsqueeze(-1))
        if mask is not None:
            h = h * mask.to(h.dtype).unsqueeze(-1)
            self_key_padding_mask = mask == 0
            if self_key_padding_mask.all(dim=1).any():
                self_key_padding_mask = self_key_padding_mask.clone()
                self_key_padding_mask[self_key_padding_mask.all(dim=1), 0] = False
        else:
            self_key_padding_mask = None

        for block_idx, block in enumerate(self.blocks):
            residual = h
            x = block["norm"](h)
            x = block["conv"](x.transpose(1, 2)).transpose(1, 2)
            x = block["mix"](x)
            h = residual + x
            self_in = block["self_norm"](h)
            self_out, _ = block["self_attn"](
                self_in,
                self_in,
                self_in,
                key_padding_mask=self_key_padding_mask,
                need_weights=False,
            )
            h = h + SELF_GATE_SCALE * self.self_gates[block_idx].tanh() * self_out
            if prompt_h is not None:
                cross_in = block["cross_norm"](h)
                cross_out, _ = block["cross_attn"](
                    cross_in,
                    prompt_h,
                    prompt_h,
                    key_padding_mask=prompt_key_padding_mask,
                    need_weights=False,
                )
                h = h + CROSS_GATE_SCALE * self.cross_gates[block_idx].tanh() * cross_out
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)

        h_before_token = h
        if TOKEN_SHARED_BLOCK:
            token_delta = TOKEN_SHARED_BLOCK_SCALE * self.token_block(h_before_token)
            self._last_token_block_norm = token_delta.detach().norm(dim=-1).mean()
            self._last_token_hidden_norm = h_before_token.detach().norm(dim=-1).mean()
            self._last_token_block_ratio = self._last_token_block_norm / self._last_token_hidden_norm.clamp_min(1e-6)
            h = h + token_delta
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)
        else:
            self._last_token_block_norm = h_before_token.detach().new_tensor(0.0)
            self._last_token_hidden_norm = h_before_token.detach().norm(dim=-1).mean()
            self._last_token_block_ratio = h_before_token.detach().new_tensor(0.0)

        hidden = self.out_norm(h)
        out = self.out_proj(hidden)
        if TOKEN_RESIDUAL_SCALE > 0:
            out = out + TOKEN_RESIDUAL_SCALE * self.token_residual_proj(hidden)
        out = clamp_velocity(out)
        self._last_velocity_out_norm = out.detach().norm(dim=-1).mean()
        self._last_out_proj_weight_norm = self.out_proj.weight.detach().norm()
        if squeeze:
            out = out.squeeze(1)
            hidden = hidden.squeeze(1)
        if return_hidden:
            return out, hidden
        return out


class AuxTokenHead(nn.Module):
    def __init__(self, hidden_dim=FLOW_HIDDEN_DIM, vocab_size=30522):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, hidden):
        return self.net(hidden)


class StartMLP(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=START_MLP_HIDDEN_DIM):
        super().__init__()
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_prompt, pos, mask=None):
        prompt_summary = z_prompt.mean(dim=1)
        prompt_summary = prompt_summary.unsqueeze(1).expand(-1, pos.size(1), -1)
        pos_h = self.pos_proj(pos.unsqueeze(-1))
        z_start = self.net(torch.cat([prompt_summary, pos_h], dim=-1))
        if mask is not None:
            z_start = z_start * mask.to(z_start.dtype).unsqueeze(-1)
        return z_start


class _StartTransformerLayer(nn.Module):
    def __init__(self, latent_dim, num_heads, ffn_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, latent_dim),
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)

    def forward(self, x, z_prompt):
        x2, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x2)
        x2, _ = self.cross_attn(x, z_prompt, z_prompt)
        x = self.norm2(x + x2)
        x = self.norm3(x + self.ffn(x))
        return x


class StartTransformer(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        num_layers=START_TRANSFORMER_LAYERS,
        num_heads=START_TRANSFORMER_HEADS,
        ffn_dim=START_TRANSFORMER_HIDDEN_DIM,
    ):
        super().__init__()
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.layers = nn.ModuleList([
            _StartTransformerLayer(latent_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_prompt, pos, mask=None):
        x = self.pos_proj(pos.unsqueeze(-1))
        for layer in self.layers:
            x = layer(x, z_prompt)
        x = self.out_proj(self.out_norm(x))
        if mask is not None:
            x = x * mask.to(x.dtype).unsqueeze(-1)
        return x


class VQLatentTokenizer(nn.Module):
    """Per-token latent VQ tokenizer for decoder-readable suffix latents."""

    def __init__(
        self,
        latent_dim=256,
        codebook_size=512,
        commitment_cost=0.25,
        use_mlp=False,
        hidden_dim=512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.encoder = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            if use_mlp
            else nn.Identity()
        )
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.decoder = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            if use_mlp
            else nn.Identity()
        )
        nn.init.normal_(self.codebook.weight, mean=TARGET_LATENT_MEAN, std=TARGET_LATENT_STD)

    def quantize_encoded(self, z_e):
        flat = z_e.reshape(-1, z_e.size(-1)).float()
        codebook = self.codebook.weight.float()
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        code_ids = distances.argmin(dim=1).reshape(z_e.shape[:-1])
        z_q = self.codebook(code_ids)
        return z_q, code_ids

    def encode(self, z, mask=None):
        z_e = self.encoder(z)
        _z_q, code_ids = self.quantize_encoded(z_e)
        if mask is not None:
            code_ids = code_ids.masked_fill(~mask.bool(), 0)
        return code_ids

    def decode_codes(self, code_ids, mask=None):
        z = self.decoder(self.codebook(code_ids))
        if mask is not None:
            z = z * mask.to(z.dtype).unsqueeze(-1)
        return z

    def forward(self, z, mask=None):
        z_e = self.encoder(z)
        z_q_raw, code_ids = self.quantize_encoded(z_e)
        z_q = self.decoder(z_q_raw)
        if mask is not None:
            valid = mask.bool().unsqueeze(-1)
            z_q = z_q * valid.to(z_q.dtype)
            z_e = z_e * valid.to(z_e.dtype)
            z = z * valid.to(z.dtype)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_e, z_q_raw.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        return z_q, code_ids, vq_loss, {
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
        }

    @torch.no_grad()
    def usage_stats(self, code_ids, mask=None):
        ids = code_ids[mask.bool()] if mask is not None else code_ids.reshape(-1)
        if ids.numel() == 0:
            return 0.0, 100.0, 0
        counts = torch.bincount(ids.reshape(-1), minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        used = int((counts > 0).sum().item())
        entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
        perplexity = float(entropy.exp().item())
        dead_pct = float(100.0 * (self.codebook_size - used) / self.codebook_size)
        return perplexity, dead_pct, used


class VQDecoderAdapter(nn.Module):
    """Small residual adapter that makes VQ/generated suffix latents easier to decode."""

    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        layers=2,
        delta_scale=0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.delta_scale = delta_scale
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        blocks = []
        for _ in range(layers):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, latent_dim),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_suffix, z_prompt, pos, mask=None, return_delta=False):
        prompt = self.prompt_proj(z_prompt.mean(dim=1)).unsqueeze(1)
        x = z_suffix + prompt + self.pos_proj(pos.unsqueeze(-1))
        for block in self.blocks:
            x = x + block(x)
            if mask is not None:
                x = x * mask.to(x.dtype).unsqueeze(-1)
        delta = self.delta_scale * self.out_proj(self.out_norm(x))
        if mask is not None:
            delta = delta * mask.to(delta.dtype).unsqueeze(-1)
        out = z_suffix + delta
        if return_delta:
            return out, delta
        return out


class SyntaxTokenRefiner(nn.Module):
    """Prompt + rough draft tokens/latents -> parallel refined suffix token logits."""

    def __init__(
        self,
        vocab_size,
        latent_dim=256,
        hidden_dim=512,
        num_layers=3,
        num_heads=8,
        mixer_layers=2,
        mixer_kernel=5,
        mixer_scale=0.5,
        pad_token_id=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.mixer_scale = mixer_scale
        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(vocab_size, latent_dim, padding_idx=pad_token_id)
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.conf_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.prompt_layers = nn.ModuleList([
            _StartTransformerLayer(latent_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.mixers = nn.ModuleList([
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(latent_dim),
                    "conv": nn.Conv1d(
                        latent_dim,
                        latent_dim,
                        kernel_size=mixer_kernel,
                        padding=mixer_kernel // 2,
                        groups=latent_dim,
                    ),
                    "gate": nn.Linear(latent_dim, latent_dim),
                    "out": nn.Linear(latent_dim, latent_dim),
                }
            )
            for _ in range(mixer_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, vocab_size)

    def forward(self, z_prompt, draft_ids, z_draft, pos, suffix_mask=None, draft_conf=None):
        if draft_conf is None:
            draft_conf = torch.ones_like(draft_ids, dtype=z_draft.dtype)
        x = (
            self.token_emb(draft_ids.clamp(0, self.vocab_size - 1))
            + self.latent_proj(z_draft)
            + self.prompt_proj(z_prompt.mean(dim=1)).unsqueeze(1)
            + self.pos_proj(pos.unsqueeze(-1))
            + self.conf_proj(draft_conf.to(z_draft.dtype).unsqueeze(-1))
        )
        key_padding_mask = None
        if suffix_mask is not None:
            key_padding_mask = ~suffix_mask.bool()
        for prompt_layer, self_layer in zip(self.prompt_layers, self.self_layers):
            x = prompt_layer(x, z_prompt)
            x = self_layer(x, src_key_padding_mask=key_padding_mask)
            if suffix_mask is not None:
                x = x * suffix_mask.to(x.dtype).unsqueeze(-1)
        for mixer in self.mixers:
            h = mixer["norm"](x)
            conv = mixer["conv"](h.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(mixer["gate"](h))
            x = x + self.mixer_scale * mixer["out"](conv * gate)
            if suffix_mask is not None:
                x = x * suffix_mask.to(x.dtype).unsqueeze(-1)
        logits = self.out_proj(self.out_norm(x))
        if suffix_mask is not None:
            logits = logits.masked_fill(~suffix_mask.bool().unsqueeze(-1), 0.0)
        return logits


class CodePrior(nn.Module):
    """Prompt-conditioned parallel categorical suffix-code predictor."""

    def __init__(
        self,
        latent_dim=256,
        codebook_size=512,
        num_layers=2,
        num_heads=8,
        ffn_dim=512,
        mixer_layers=2,
        mixer_kernel=5,
        mixer_scale=0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.mixer_scale = mixer_scale
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.layers = nn.ModuleList([
            _StartTransformerLayer(latent_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.mixers = nn.ModuleList([
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(latent_dim),
                    "conv": nn.Conv1d(
                        latent_dim,
                        latent_dim,
                        kernel_size=mixer_kernel,
                        padding=mixer_kernel // 2,
                        groups=latent_dim,
                    ),
                    "gate": nn.Linear(latent_dim, latent_dim),
                    "out": nn.Linear(latent_dim, latent_dim),
                }
            )
            for _ in range(mixer_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, codebook_size)

    def forward(self, z_prompt, pos, mask=None, return_hidden=False):
        x = self.pos_proj(pos.unsqueeze(-1))
        for layer in self.layers:
            x = layer(x, z_prompt)
        for mixer in self.mixers:
            h = mixer["norm"](x)
            conv = mixer["conv"](h.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(mixer["gate"](h))
            x = x + self.mixer_scale * mixer["out"](conv * gate)
            if mask is not None:
                x = x * mask.to(x.dtype).unsqueeze(-1)
        logits = self.out_proj(self.out_norm(x))
        if mask is not None:
            logits = logits.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        if return_hidden:
            return logits, x
        return logits


class HierCodePrior(nn.Module):
    """Prompt -> plan slots -> parallel suffix code logits."""

    def __init__(
        self,
        latent_dim=256,
        codebook_size=512,
        plan_slots=8,
        num_layers=2,
        num_heads=8,
        ffn_dim=512,
        mixer_layers=2,
        mixer_kernel=5,
        mixer_scale=0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.plan_slots = plan_slots
        self.mixer_scale = mixer_scale
        self.plan_queries = nn.Parameter(torch.randn(plan_slots, latent_dim) * 0.02)
        self.plan_layers = nn.ModuleList([
            _StartTransformerLayer(latent_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.token_layers = nn.ModuleList([
            _StartTransformerLayer(latent_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.plan_cross = nn.ModuleList([
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(latent_dim),
                    "attn": nn.MultiheadAttention(latent_dim, num_heads, batch_first=True),
                    "ff": nn.Sequential(
                        nn.LayerNorm(latent_dim),
                        nn.Linear(latent_dim, ffn_dim),
                        nn.GELU(),
                        nn.Linear(ffn_dim, latent_dim),
                    ),
                }
            )
            for _ in range(num_layers)
        ])
        self.mixers = nn.ModuleList([
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(latent_dim),
                    "conv": nn.Conv1d(
                        latent_dim,
                        latent_dim,
                        kernel_size=mixer_kernel,
                        padding=mixer_kernel // 2,
                        groups=latent_dim,
                    ),
                    "gate": nn.Linear(latent_dim, latent_dim),
                    "out": nn.Linear(latent_dim, latent_dim),
                }
            )
            for _ in range(mixer_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, codebook_size)

    def forward(self, z_prompt, pos, mask=None, return_plans=False, return_hidden=False):
        batch = z_prompt.size(0)
        plans = self.plan_queries.unsqueeze(0).expand(batch, -1, -1)
        for layer in self.plan_layers:
            plans = layer(plans, z_prompt)

        x = self.pos_proj(pos.unsqueeze(-1))
        for layer, plan_block in zip(self.token_layers, self.plan_cross):
            x = layer(x, z_prompt)
            plan_in = plan_block["norm"](x)
            plan_out, _ = plan_block["attn"](plan_in, plans, plans, need_weights=False)
            x = x + plan_out
            x = x + plan_block["ff"](x)
            if mask is not None:
                x = x * mask.to(x.dtype).unsqueeze(-1)

        for mixer in self.mixers:
            h = mixer["norm"](x)
            conv = mixer["conv"](h.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(mixer["gate"](h))
            x = x + self.mixer_scale * mixer["out"](conv * gate)
            if mask is not None:
                x = x * mask.to(x.dtype).unsqueeze(-1)

        logits = self.out_proj(self.out_norm(x))
        if mask is not None:
            logits = logits.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        if return_plans and return_hidden:
            return logits, plans, x
        if return_plans:
            return logits, plans
        if return_hidden:
            return logits, x
        return logits


class RouteCodePrior(HierCodePrior):
    """Prompt -> route hubs -> smooth slot routing -> parallel suffix code logits."""

    def __init__(self, *args, route_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.route_scale = route_scale
        self.route_norm = nn.LayerNorm(self.latent_dim)
        self.route_proj = nn.Linear(self.latent_dim, self.plan_slots)
        self.route_value_proj = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, z_prompt, pos, mask=None, return_aux=False, return_hidden=False):
        code_logits, plans, hidden = super().forward(
            z_prompt,
            pos,
            mask=mask,
            return_plans=True,
            return_hidden=True,
        )
        route_logits = self.route_proj(self.route_norm(hidden))
        route_probs = torch.softmax(route_logits, dim=-1)
        routed = route_probs @ self.route_value_proj(plans)
        hidden = hidden + self.route_scale * routed
        if mask is not None:
            hidden = hidden * mask.to(hidden.dtype).unsqueeze(-1)
            route_logits = route_logits.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        code_logits = self.out_proj(self.out_norm(hidden))
        if mask is not None:
            code_logits = code_logits.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        aux = {
            "plans": plans,
            "route_logits": route_logits,
            "route_probs": route_probs,
        }
        if return_aux and return_hidden:
            return code_logits, aux, hidden
        if return_aux:
            return code_logits, aux
        if return_hidden:
            return code_logits, hidden
        return code_logits


class MaskedCodeRefiner(nn.Module):
    """Prompt + partially known VQ codes -> parallel refined code logits."""

    def __init__(
        self,
        latent_dim=256,
        codebook_size=512,
        num_layers=2,
        num_heads=8,
        ffn_dim=512,
        mixer_layers=2,
        mixer_kernel=5,
        mixer_scale=0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.mixer_scale = mixer_scale
        self.mask_code = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.known_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.prompt_layers = nn.ModuleList([
            _StartTransformerLayer(latent_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.mixers = nn.ModuleList([
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(latent_dim),
                    "conv": nn.Conv1d(
                        latent_dim,
                        latent_dim,
                        kernel_size=mixer_kernel,
                        padding=mixer_kernel // 2,
                        groups=latent_dim,
                    ),
                    "gate": nn.Linear(latent_dim, latent_dim),
                    "out": nn.Linear(latent_dim, latent_dim),
                }
            )
            for _ in range(mixer_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, codebook_size)

    def forward(self, z_prompt, code_emb, known_mask, pos, suffix_mask=None, return_hidden=False):
        known = known_mask.to(code_emb.dtype).unsqueeze(-1)
        mask_code = self.mask_code.to(code_emb.dtype).view(1, 1, -1)
        x = known * code_emb + (1.0 - known) * mask_code
        x = x + self.pos_proj(pos.unsqueeze(-1)) + self.known_proj(known)
        key_padding_mask = None
        if suffix_mask is not None:
            key_padding_mask = ~suffix_mask.bool()
        for prompt_layer, self_layer in zip(self.prompt_layers, self.self_layers):
            x = prompt_layer(x, z_prompt)
            x = self_layer(x, src_key_padding_mask=key_padding_mask)
            if suffix_mask is not None:
                x = x * suffix_mask.to(x.dtype).unsqueeze(-1)
        for mixer in self.mixers:
            h = mixer["norm"](x)
            conv = mixer["conv"](h.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(mixer["gate"](h))
            x = x + self.mixer_scale * mixer["out"](conv * gate)
            if suffix_mask is not None:
                x = x * suffix_mask.to(x.dtype).unsqueeze(-1)
        logits = self.out_proj(self.out_norm(x))
        if suffix_mask is not None:
            logits = logits.masked_fill(~suffix_mask.bool().unsqueeze(-1), 0.0)
        if return_hidden:
            return logits, x
        return logits


class DenoisingPrior(nn.Module):
    """
    x0-prediction denoising prior.
    Input : z_t (noisy suffix latents), z_prompt, alpha (noise level), pos
    Output: pred_z_real  (residual on z_t, zero-init)
    """

    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.alpha_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.layers = nn.ModuleList([
            _StartTransformerLayer(latent_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t, z_prompt, alpha, pos, mask=None):
        # alpha: [B] — one scalar noise level per sample
        alpha_emb = self.alpha_embed(alpha.unsqueeze(-1)).unsqueeze(1).expand(-1, z_t.size(1), -1)
        pos_emb = self.pos_proj(pos.unsqueeze(-1))
        x = z_t + alpha_emb + pos_emb
        for layer in self.layers:
            x = layer(x, z_prompt)
        pred = z_t + self.out_proj(self.out_norm(x))  # residual: at init pred == z_t
        if mask is not None:
            pred = pred * mask.to(pred.dtype).unsqueeze(-1)
        return pred


class DenoisingPriorSampler(nn.Module):
    """
    Wraps DenoisingPrior to expose the start_mlp(z_prompt, pos, mask) interface.

    use_oracle=True  (stage2 training with oracle z_t):
        set_oracle_target(z_real) before forward() →
        z_t = alpha * z_real + beta * noise  (matches prior oracle training distribution)
    use_oracle=False (inference-matched training / actual inference):
        set_oracle_target() is a no-op; forward() runs the chain:
        z = noise → prior(chain_alphas[0]) → … → prior(chain_alphas[-1])
    """

    def __init__(self, prior, latent_dim=256, alpha=0.5,
                 chain_alphas=None, use_oracle=True):
        super().__init__()
        self.prior = prior
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.chain_alphas = chain_alphas if chain_alphas is not None else [0.3, 0.5, 0.7]
        self.use_oracle = use_oracle
        self._oracle_target = None

    def set_oracle_target(self, z_target):
        """Set before forward() to use oracle z_t.
        No-op when use_oracle=False — sampler always runs chain mode."""
        if self.use_oracle:
            self._oracle_target = z_target

    def forward(self, z_prompt, pos, mask=None):
        B, T = z_prompt.size(0), pos.size(1)

        if self._oracle_target is not None:
            # Oracle mode: single denoising step from alpha-noised real latent
            alpha_val = self.alpha
            beta = (1.0 - alpha_val ** 2) ** 0.5
            z_t = alpha_val * self._oracle_target[:B].detach() + beta * torch.randn(
                B, T, self.latent_dim, device=z_prompt.device, dtype=z_prompt.dtype
            )
            self._oracle_target = None
            if mask is not None:
                z_t = z_t * mask.to(z_t.dtype).unsqueeze(-1)
            alpha_t = z_prompt.new_full((B,), alpha_val)
            return self.prior(z_t, z_prompt, alpha_t, pos, mask)
        else:
            # Chain mode: pure noise → prior(chain_alphas[0]) → … → prior(chain_alphas[-1])
            z = TARGET_LATENT_STD * torch.randn(
                B, T, self.latent_dim, device=z_prompt.device, dtype=z_prompt.dtype
            ) + TARGET_LATENT_MEAN
            if mask is not None:
                z = z * mask.to(z.dtype).unsqueeze(-1)
            for alpha_val in self.chain_alphas:
                alpha_t = z_prompt.new_full((B,), alpha_val)
                z = self.prior(z, z_prompt, alpha_t, pos, mask)
            return z


class DraftPriorSampler(DenoisingPriorSampler):
    """
    Draft-conditioned prior sampler.

    Training/inference path:
        set_draft_target(z_draft) before forward() →
        z_t = alpha * z_draft + beta * noise → prior(z_t, z_prompt, alpha, pos)

    It keeps DenoisingPriorSampler's fallback behavior so old call sites remain safe,
    but stage2 should provide draft targets when DRAFT_PRIOR is enabled.
    """

    def __init__(self, prior, latent_dim=256, alpha=0.7):
        super().__init__(prior, latent_dim=latent_dim, alpha=alpha, use_oracle=False)
        self._draft_target = None

    def set_draft_target(self, z_draft):
        self._draft_target = z_draft

    def forward(self, z_prompt, pos, mask=None):
        if self._draft_target is None:
            return super().forward(z_prompt, pos, mask)

        B, T = z_prompt.size(0), pos.size(1)
        z_draft = self._draft_target[:B].detach()
        self._draft_target = None
        alpha_val = self.alpha
        beta = (1.0 - alpha_val ** 2) ** 0.5
        z_t = alpha_val * z_draft + beta * (
            torch.randn(B, T, self.latent_dim, device=z_prompt.device, dtype=z_prompt.dtype)
            * TARGET_LATENT_STD
            + TARGET_LATENT_MEAN
        )
        if mask is not None:
            z_t = z_t * mask.to(z_t.dtype).unsqueeze(-1)
        alpha_t = z_prompt.new_full((B,), alpha_val)
        return self.prior(z_t, z_prompt, alpha_t, pos, mask)


def flow_token_block_stats(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)

    def scalar(name):
        value = getattr(model, name, None)
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            return value.detach().float().item()
        return float(value)

    return {
        "token_block_norm": scalar("_last_token_block_norm"),
        "token_block_ratio": scalar("_last_token_block_ratio"),
        "token_hidden_norm": scalar("_last_token_hidden_norm"),
        "velocity_out_norm": scalar("_last_velocity_out_norm"),
        "out_proj_weight_norm": scalar("_last_out_proj_weight_norm"),
    }


class MetricNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=METRIC_HIDDEN_DIM, log_bound=METRIC_LOG_BOUND):
        super().__init__()
        self.log_bound = log_bound
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, t, z_cond, pos):
        inp = torch.cat([z_t, z_cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        log_g = self.net(inp)
        log_g = log_g - log_g.mean(dim=-1, keepdim=True)
        log_g = self.log_bound * torch.tanh(log_g / self.log_bound)
        g_diag = torch.exp(log_g)
        return g_diag / g_diag.mean(dim=-1, keepdim=True).clamp_min(1e-6)


class LatentProjector(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        depth=3,
        residual_scale=0.10,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(latent_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_gen, z_prompt, mask=None):
        prompt = z_prompt.mean(dim=1)
        prompt_h = self.prompt_proj(prompt).unsqueeze(1).expand(-1, z_gen.size(1), -1)
        h = self.in_proj(torch.cat([z_gen, prompt.unsqueeze(1).expand_as(z_gen)], dim=-1))
        h = h + prompt_h
        for block in self.blocks:
            h = h + block(h)
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)
        raw_delta = self.out_proj(self.out_norm(h))
        delta = self.residual_scale * torch.tanh(raw_delta)
        if mask is not None:
            delta = delta * mask.to(delta.dtype).unsqueeze(-1)
        return z_gen + delta, delta


class ResidualRefiner(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        depth=2,
        residual_scale=0.03,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
        )
        self.in_proj = nn.Linear(latent_dim * 2 + 1, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, z_prompt, pos, mask=None):
        prompt = z_prompt.mean(dim=1).unsqueeze(1).expand_as(z)
        prompt_h = self.prompt_proj(prompt)
        h = self.in_proj(torch.cat([z, prompt, pos.unsqueeze(-1)], dim=-1)) + prompt_h
        for block in self.blocks:
            h = h + block(h)
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)
        raw_delta = self.out_proj(self.out_norm(h))
        delta = self.residual_scale * torch.tanh(raw_delta)
        if mask is not None:
            delta = delta * mask.to(delta.dtype).unsqueeze(-1)
        return z + delta, delta


def prompt_condition(z_data, attention_mask, prompt_len=PROMPT_LEN):
    prompt_z = z_data[:, :prompt_len, :]
    prompt_mask = attention_mask[:, :prompt_len].to(prompt_z.dtype).unsqueeze(-1)
    return prompt_z * prompt_mask


def suffix_positions(batch_size, suffix_len, device, dtype=torch.float32):
    pos = torch.arange(PROMPT_LEN, PROMPT_LEN + suffix_len, device=device, dtype=dtype)
    pos = pos / max(MAX_SEQ_LEN - 1, 1)
    return pos.unsqueeze(0).expand(batch_size, suffix_len)


def attention_gate_stats(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)
    return (
        SELF_GATE_SCALE * torch.stack([gate.detach().tanh().abs() for gate in model.self_gates]).mean().item(),
        CROSS_GATE_SCALE * torch.stack([gate.detach().tanh().abs() for gate in model.cross_gates]).mean().item(),
    )


def attention_gate_regularizer(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)
    self_reg = torch.stack([gate.tanh().pow(2) for gate in model.self_gates]).mean()
    cross_reg = torch.stack([gate.tanh().pow(2) for gate in model.cross_gates]).mean()
    return GATE_REG_WEIGHT * (self_reg + cross_reg)


def attention_gate_parameters(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)
    return list(model.self_gates.parameters()) + list(model.cross_gates.parameters())


def non_gate_flow_parameters(flow_net):
    gate_param_ids = {id(param) for param in attention_gate_parameters(flow_net)}
    return [param for param in flow_net.parameters() if id(param) not in gate_param_ids]


def attention_gate_grad_stats(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)

    def mean_abs_grad(gates):
        grads = [gate.grad.detach().abs().mean() for gate in gates if gate.grad is not None]
        if not grads:
            return 0.0
        return torch.stack(grads).mean().item()

    return mean_abs_grad(model.self_gates), mean_abs_grad(model.cross_gates)


def calibrate_latents(z, mask=None, target_mean=TARGET_LATENT_MEAN, target_std=TARGET_LATENT_STD, eps=1e-6):
    if not CALIBRATE_GENERATED_LATENTS:
        return z
    if mask is not None:
        valid = mask.bool()
        if valid.any():
            mean = z[valid].mean()
            std = z[valid].std().clamp_min(eps)
        else:
            mean = z.mean()
            std = z.std().clamp_min(eps)
    else:
        mean = z.mean()
        std = z.std().clamp_min(eps)
    return (z - mean) * (target_std / std) + target_mean


def natural_velocity(flow_net, metric_net, z, t, z_cond, pos):
    v = flow_net(z, t, z_cond, pos)
    pooled_cond = z_cond.mean(dim=1).unsqueeze(1).expand_as(z)
    g = metric_net(
        z.reshape(-1, z.size(-1)),
        t.reshape(-1),
        pooled_cond.reshape(-1, z.size(-1)),
        pos.reshape(-1),
    ).reshape_as(z)
    return clamp_velocity(v / g.clamp_min(1e-3)), g


def structured_target_start(z_target, mask=None, alpha=STRUCTURED_START_ALPHA):
    noise = torch.randn_like(z_target) * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    beta = max(0.0, 1.0 - alpha * alpha) ** 0.5
    z = alpha * z_target + beta * noise
    if mask is not None:
        z = z * mask.to(z.dtype).unsqueeze(-1)
    return z


def generate_suffix(
    flow_net,
    metric_net,
    z_cond,
    batch_size,
    suffix_len,
    latent_dim,
    device,
    steps=ODE_STEPS,
    mask=None,
    start_mlp=None,
    z_target_start=None,
):
    pos = suffix_positions(batch_size, suffix_len, device)
    if STRUCTURED_TARGET_START and z_target_start is not None:
        z = structured_target_start(z_target_start, mask)
    elif start_mlp is not None:
        z = start_mlp(z_cond, pos, mask)
        z = z + (START_NOISE_STD_FRAC * TARGET_LATENT_STD) * torch.randn_like(z)
    else:
        z = torch.randn(batch_size, suffix_len, latent_dim, device=device) * BASE_NOISE_STD
    if mask is not None:
        z = z * mask.to(z.dtype).unsqueeze(-1)
    z_initial = z.clone()
    dt = 1.0 / steps
    metric_snapshot = None
    for i in range(steps):
        t = torch.full((batch_size, suffix_len), i / steps, device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            v, metric_snapshot = natural_velocity(flow_net, metric_net, z, t, z_cond, pos)
        z = z + FLOW_REFINE_SCALE * v * dt
        if mask is not None:
            z = z * mask.to(z.dtype).unsqueeze(-1)
    z_uncalibrated = z.clone()
    z = calibrate_latents(z, mask)
    return z, metric_snapshot, z_initial, z_uncalibrated
