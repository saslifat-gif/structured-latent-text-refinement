"""PCA-subspace latent alignment helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def load_pca_basis(path, latent_dim, dims=64, device=None):
    """Load a PCA mean/components pair and return tensors shaped [D], [D, K]."""

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, torch.Tensor):
        mean = torch.zeros(latent_dim)
        components = ckpt
    elif isinstance(ckpt, dict):
        mean = _first_present(ckpt, ("mean", "pca_mean", "latent_mean", "mu"))
        components = _first_present(
            ckpt,
            ("components", "pca_components", "basis", "V", "V64", "principal_components"),
        )
        if components is None and "pca" in ckpt and isinstance(ckpt["pca"], dict):
            mean = mean if mean is not None else _first_present(ckpt["pca"], ("mean", "pca_mean", "latent_mean", "mu"))
            components = _first_present(
                ckpt["pca"],
                ("components", "pca_components", "basis", "V", "V64", "principal_components"),
            )
        if components is None:
            raise ValueError(
                f"PCA checkpoint {path} does not contain components. "
                "Expected one of: components, pca_components, basis, V, V64."
            )
        if mean is None:
            mean = torch.zeros(latent_dim)
    else:
        raise TypeError(f"Unsupported PCA checkpoint type in {path}: {type(ckpt).__name__}")

    mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
    components = torch.as_tensor(components, dtype=torch.float32)
    if components.ndim != 2:
        raise ValueError(f"PCA components must be 2D, got shape {tuple(components.shape)}")
    if mean.numel() != latent_dim:
        raise ValueError(f"PCA mean has {mean.numel()} dims, expected latent_dim={latent_dim}")
    if components.shape[0] == latent_dim:
        basis = components
    elif components.shape[1] == latent_dim:
        basis = components.t()
    else:
        raise ValueError(
            f"PCA components shape {tuple(components.shape)} does not match latent_dim={latent_dim}"
        )
    k = min(int(dims), basis.shape[1])
    if k <= 0:
        raise ValueError(f"pca dims must be positive, got {dims}")
    mean = mean.to(device=device)
    basis = basis[:, :k].contiguous().to(device=device)
    return mean, basis


def pca_subspace_mse(z_pred, z_target, mean, basis, mask=None):
    z_pred_low = (z_pred.float() - mean) @ basis
    z_target_low = (z_target.detach().float() - mean) @ basis
    if mask is not None:
        valid = mask.bool()
        if valid.any():
            return F.mse_loss(z_pred_low[valid], z_target_low[valid])
    return F.mse_loss(z_pred_low, z_target_low)
