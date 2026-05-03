"""World Model-Conditioned Entropy Regularized Clipping utilities.

This module adapts the WMC-ERC behavior used in OpenTinker to the
AgentGym-RL training loop. It uses per-token policy entropy as a world-model
uncertainty signal on observation tokens and applies a turn-level mask or
soft clipping coefficient to actor advantages.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist


def compute_turn_boundaries(response_mask: torch.Tensor) -> List[List[Tuple[int, int]]]:
    """Extract contiguous assistant-token spans from response_mask.

    Each span corresponds to one action turn in the rollout response region.
    """
    boundaries_per_sample: List[List[Tuple[int, int]]] = []

    for sample_mask in response_mask.bool():
        sample_boundaries: List[Tuple[int, int]] = []
        start = None
        for idx, flag in enumerate(sample_mask.tolist()):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                sample_boundaries.append((start, idx))
                start = None
        if start is not None:
            sample_boundaries.append((start, len(sample_mask)))
        boundaries_per_sample.append(sample_boundaries)

    return boundaries_per_sample


def compute_s_star(
    old_log_probs: torch.Tensor,
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn policy blind confidence S_*.

    S_*^t = mean over action tokens in turn t of: p_k * (H + log p_k)
    """
    batch_size = old_log_probs.shape[0]
    device = old_log_probs.device
    s_star_per_sample = []

    for i in range(batch_size):
        s_star_turns = []
        for start, end in turn_boundaries[i]:
            log_p = old_log_probs[i, start:end]
            H = entropys[i, start:end]
            mask = response_mask[i, start:end]
            count = mask.sum()

            if count > 0:
                p_k = torch.exp(log_p)
                s_token = p_k * (H + log_p)
                s_token = torch.nan_to_num(s_token, nan=0.0)
                s_t = (s_token * mask).sum() / count
            else:
                s_t = torch.tensor(0.0, device=device)

            s_star_turns.append(s_t)
        s_star_per_sample.append(s_star_turns)

    return s_star_per_sample


def compute_h_wm(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask_response: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn World Model uncertainty H_WM.

    H_WM^t = mean prediction entropy at env token positions following action turn t.
    """
    batch_size = entropys.shape[0]
    seq_len = entropys.shape[1]
    device = entropys.device
    env_mask = attention_mask_response * (1.0 - response_mask)

    h_wm_per_sample = []

    for i in range(batch_size):
        boundaries = turn_boundaries[i]
        h_wm_turns = []

        for t, (start, end) in enumerate(boundaries):
            if t + 1 < len(boundaries):
                env_end = boundaries[t + 1][0]
            else:
                env_end = seq_len

            region_mask = env_mask[i, end:env_end]
            region_entropy = entropys[i, end:env_end]
            count = region_mask.sum()

            if count > 0:
                h_wm_t = (region_entropy * region_mask).sum() / count
            else:
                h_wm_t = torch.tensor(0.0, device=device)

            h_wm_turns.append(h_wm_t)

        h_wm_per_sample.append(h_wm_turns)

    return h_wm_per_sample


def compute_dynamic_mask(
    s_star_per_sample: List[List[torch.Tensor]],
    h_wm_per_sample: List[List[torch.Tensor]],
    mu_base: float,
    mu_exp: float,
    eta_wm: float,
    lambda_wm: float,
    s_bar: float,
    sigma: float,
    clipping_method: str = "mask",
    h_bar: float = None,
) -> List[List[float]]:
    """Compute per-turn dynamic entropy mask or clipping coefficient."""
    mask_per_sample: List[List[float]] = []

    for sample_idx in range(len(s_star_per_sample)):
        sample_masks: List[float] = []
        for turn_idx in range(len(s_star_per_sample[sample_idx])):
            s_t = s_star_per_sample[sample_idx][turn_idx].detach().item()
            world_model_loss = h_wm_per_sample[sample_idx][turn_idx].detach().item()

            h_factor = eta_wm * np.exp(-lambda_wm * world_model_loss)

            if s_t > s_bar:
                threshold = mu_base * h_factor * sigma
                diff = s_t - s_bar
            else:
                threshold = mu_exp * h_factor * sigma
                diff = s_bar - s_t

            if clipping_method == "mask":
                m_t = 1.0 if diff <= threshold else 0.0
            elif clipping_method == "sigmoid":
                dh = np.clip(world_model_loss - h_bar, -2.0, 2.0)
                tau = sigma * 0.2 + 1e-8
                delta = s_t - s_bar
                if delta > 0:
                    width = mu_base * np.exp(-lambda_wm * dh) * sigma
                    m_t = 1.0 / (1.0 + np.exp(max(-500, min(500, (delta - width) / tau))))
                else:
                    width = mu_exp * np.exp(lambda_wm * dh) * sigma
                    m_t = 1.0 / (1.0 + np.exp(max(-500, min(500, (-delta - width) / tau))))
            elif clipping_method == "gaussian":
                dh = np.clip(world_model_loss - h_bar, -2.0, 2.0)
                delta = s_t - s_bar
                if delta > 0:
                    width = mu_base * np.exp(-lambda_wm * dh) * sigma
                else:
                    width = mu_exp * np.exp(lambda_wm * dh) * sigma
                m_t = np.exp(-0.5 * delta ** 2 / (width ** 2 + 1e-8))
            else:
                m_t = min(1.0, threshold / (diff + 1e-8))

            sample_masks.append(m_t)
        mask_per_sample.append(sample_masks)

    return mask_per_sample


def apply_wmc_erc(
    batch,
    entropys: torch.Tensor,
    wmc_erc_config,
    running_stats: Dict[str, float],
):
    """Apply WMC-ERC dynamic entropy clipping to batch advantages."""
    enable = wmc_erc_config.get("enable", True) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "enable", True)
    if not enable:
        return batch, {}

    clipping_type = wmc_erc_config.get("clipping_type", "batch") if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clipping_type", "batch")
    clipping_method = wmc_erc_config.get("clipping_method", "mask") if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clipping_method", "mask")
    clip_positive_only = wmc_erc_config.get("clip_positive_only", False) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clip_positive_only", False)
    inverse_sft_mask = wmc_erc_config.get("inverse_sft_mask", False) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "inverse_sft_mask", False)

    response_mask = batch.batch["response_mask"]
    old_log_probs = batch.batch["old_log_probs"]
    advantages = batch.batch["advantages"]
    batch_size = advantages.shape[0]
    response_length = advantages.shape[1]
    attention_mask = batch.batch["attention_mask"]
    attention_mask_response = attention_mask[:, -response_length:]

    turn_boundaries = compute_turn_boundaries(response_mask)

    s_star = compute_s_star(old_log_probs, entropys, response_mask, turn_boundaries)
    h_wm = compute_h_wm(entropys, response_mask, attention_mask_response, turn_boundaries)

    all_s = [s.item() for turns in s_star for s in turns]
    all_h = [h.item() for turns in h_wm for h in turns]

    if not all_s:
        return batch, {}

    batch_s_bar = np.mean(all_s)
    batch_s_std = np.std(all_s) + 1e-8
    batch_h_bar = np.mean(all_h) + 1e-8

    momentum = wmc_erc_config.get("momentum", 0.9) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "momentum", 0.9)
    if len(running_stats.keys()) == 0:
        running_stats["s_bar"] = batch_s_bar
        running_stats["s_std"] = batch_s_std
        running_stats["h_bar"] = batch_h_bar
    else:
        running_stats["s_bar"] = (1 - momentum) * batch_s_bar + momentum * running_stats["s_bar"]
        running_stats["s_std"] = (1 - momentum) * batch_s_std + momentum * running_stats["s_std"]
        running_stats["h_bar"] = (1 - momentum) * batch_h_bar + momentum * running_stats["h_bar"]

    if clipping_type == "global":
        use_s_bar = running_stats["s_bar"]
        use_s_std = running_stats["s_std"]
    else:
        use_s_bar = batch_s_bar
        use_s_std = batch_s_std

    if clipping_method == "add":
        # Additive curiosity bonus: A' = A + coef * max(0, z(wm_loss_per_turn))
        wmloss_add_coef = float(wmc_erc_config.get("wmloss_add_coef", 0.1))
        
        all_h_tensor = torch.tensor(all_h, device=advantages.device, dtype=torch.float32)
        mu_h = all_h_tensor.mean()
        sg_h = all_h_tensor.std(unbiased=False).clamp(min=1e-3)
        
        if dist.is_available() and dist.is_initialized():
            stats = torch.stack([mu_h, sg_h])
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            mu_h, sg_h = stats[0], stats[1].clamp(min=1e-3)
            
        offsets_per_sample = []
        all_offsets = []
        
        for i in range(batch_size):
            sample_offsets = []
            for t in range(len(h_wm[i])):
                h_t = h_wm[i][t]
                z_t = ((h_t - mu_h) / sg_h).clamp(-3.0, 3.0)
                # z_t = torch.clamp(z_t, min=0.0)
                offset_t = wmloss_add_coef * z_t
                sample_offsets.append(offset_t)
                
                # Apply to advantages
                start, end = turn_boundaries[i][t]
                advantages[i, start:end] += offset_t
                all_offsets.append(offset_t.item())
            offsets_per_sample.append(sample_offsets)
            
        batch.batch["advantages"] = advantages
        
        # Collect add-specific metrics
        add_metrics = {
            "wmc_erc/wmloss_offset_mean": float(np.mean(all_offsets)) if all_offsets else 0.0,
            "wmc_erc/wmloss_offset_std": float(np.std(all_offsets)) if all_offsets else 0.0,
            "wmc_erc/wmloss_offset_max": float(np.max(all_offsets)) if all_offsets else 0.0,
            "wmc_erc/wmloss_offset_min": float(np.min(all_offsets)) if all_offsets else 0.0,
        }
    else:
        # Standard multiplicative/masking modes
        mask = compute_dynamic_mask(
            s_star, h_wm, mu_base, mu_exp, eta_wm, lambda_wm,
            s_bar=use_s_bar,
            sigma=use_s_std,
            clipping_method=clipping_method,
            h_bar=running_stats["h_bar"],
        )

        for i in range(batch_size):
            for t, (start, end) in enumerate(turn_boundaries[i]):
                if t < len(mask[i]):
                    m_t = mask[i][t]

                    if inverse_sft_mask:
                        if t + 1 < len(turn_boundaries[i]):
                            env_end = turn_boundaries[i][t + 1][0]
                        else:
                            env_end = response_length

                        if m_t == 1.0 and clipping_method != "mask":
                            sft_weight = 1.0
                        else:
                            if m_t > 0.7:
                                sft_weight = 1.0 - m_t
                            else:
                                sft_weight = 2.0 - m_t
                        region_mask = env_mask[i, end:env_end]
                        sft_weights[i, end:env_end] = region_mask * sft_weight

                    if m_t < 1.0:
                        if clip_positive_only:
                            turn_adv = advantages[i, start:end]
                            advantages[i, start:end] = torch.where(turn_adv > 0, turn_adv * m_t, turn_adv)
                        else:
                            advantages[i, start:end] *= m_t
        batch.batch["advantages"] = advantages
        add_metrics = {}

    if inverse_sft_mask:
        batch.batch["sft_weights"] = sft_weights

    if clipping_method != "add":
        all_m = [m for turns in mask for m in turns]
        num_collapsing_violated = 0
        num_exploration_violated = 0
        for i in range(len(s_star)):
            for t in range(len(s_star[i])):
                if mask[i][t] < 1.0:
                    if s_star[i][t].item() > use_s_bar:
                        num_collapsing_violated += 1
                    else:
                        num_exploration_violated += 1
    else:
        all_m = [1.0] # Dummy for add mode
        num_collapsing_violated = 0
        num_exploration_violated = 0

    env_mask = attention_mask_response * (1.0 - response_mask)
    env_count = env_mask.sum()
    wm_nll = (-(old_log_probs * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0

    metrics = {
        "wmc_erc/batch_s_bar": float(batch_s_bar),
        "wmc_erc/batch_s_std": float(batch_s_std),
        "wmc_erc/batch_h_bar": float(batch_h_bar),
        "wmc_erc/running_s_bar": float(running_stats["s_bar"]),
        "wmc_erc/running_s_std": float(running_stats["s_std"]),
        "wmc_erc/running_h_bar": float(running_stats["h_bar"]),
        "wmc_erc/mask_ratio": float(np.mean(all_m)) if all_m else 1.0,
        "wmc_erc/num_violated_turns": sum(1 for m in all_m if m < 1.0),
        "wmc_erc/num_collapsing_violated": num_collapsing_violated,
        "wmc_erc/num_exploration_violated": num_exploration_violated,
        "wmc_erc/total_turns": len(all_m) if clipping_method != "add" else len(all_h),
        "wmc_erc/wm_nll": wm_nll,
    }
    metrics.update(add_metrics)

    return batch, metrics
