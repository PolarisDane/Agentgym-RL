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
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask_response: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn World Model Loss (NLL).

    WM_Loss^t = mean negative log-likelihood at env token positions following action turn t.
    """
    batch_size = old_log_probs.shape[0]
    seq_len = old_log_probs.shape[1]
    device = old_log_probs.device
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
            region_log_prob = old_log_probs[i, end:env_end]
            count = region_mask.sum()

            if count > 0:
                # WM Loss is -log_prob
                h_wm_t = -(region_log_prob * region_mask).sum() / count
            else:
                h_wm_t = torch.tensor(0.0, device=device)

            h_wm_turns.append(h_wm_t)

        h_wm_per_sample.append(h_wm_turns)

    return h_wm_per_sample


def compute_h_wm_entropy(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask_response: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn World Model Entropy.

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


def compute_h_action(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn mean action entropy.

    H_action^t = mean per-token entropy over action tokens in turn t.
    """
    batch_size = entropys.shape[0]
    device = entropys.device
    h_action_per_sample = []

    for i in range(batch_size):
        h_action_turns = []
        for start, end in turn_boundaries[i]:
            H = entropys[i, start:end]
            mask = response_mask[i, start:end]
            count = mask.sum()

            if count > 0:
                h_t = (H * mask).sum() / count
            else:
                h_t = torch.tensor(0.0, device=device)

            h_action_turns.append(h_t)
        h_action_per_sample.append(h_action_turns)

    return h_action_per_sample


def compute_pi_per_turn(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn action probability p(a|s).

    p(a|s)^t = exp(sum_{k in turn t} log p_k)
    """
    batch_size = old_log_probs.shape[0]
    device = old_log_probs.device
    pi_per_turn_per_sample = []

    for i in range(batch_size):
        pi_turns = []
        for start, end in turn_boundaries[i]:
            log_p = old_log_probs[i, start:end]
            mask = response_mask[i, start:end]

            if mask.sum() > 0:
                # Sum log probs of action tokens in this turn
                turn_log_prob = (log_p * mask).sum()
                pi_t = torch.exp(turn_log_prob)
            else:
                pi_t = torch.tensor(1.0, device=device)

            pi_turns.append(pi_t)
        pi_per_turn_per_sample.append(pi_turns)

    return pi_per_turn_per_sample


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
    step: int = None,
):
    """Apply WMC-ERC dynamic entropy clipping to batch advantages."""
    enable = wmc_erc_config.get("enable", True) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "enable", True)
    if not enable:
        return batch, {}

    if step is not None:
        running_stats['step'] = step
    else:
        running_stats['step'] = running_stats.get('step', 0) + 1
    current_step = running_stats['step']

    clipping_type = wmc_erc_config.get("clipping_type", "batch") if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clipping_type", "batch")
    clipping_method = wmc_erc_config.get("clipping_method", "mask") if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clipping_method", "mask")
    clip_positive_only = wmc_erc_config.get("clip_positive_only", False) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clip_positive_only", False)
    inverse_sft_mask = wmc_erc_config.get("inverse_sft_mask", False) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "inverse_sft_mask", False)

    # Success/Failure info
    wmloss_add_only_failed = bool(wmc_erc_config.get("wmloss_add_only_failed", False))
    if wmloss_add_only_failed and 'task_scores' in batch.batch.keys():
        # Sum scores over tokens/turns to get trajectory success
        traj_scores = batch.batch['task_scores'].sum(dim=-1) # [B]
        traj_failed = (traj_scores <= 0.0) # only for alfworld, not for sciworld
    else:
        traj_failed = torch.ones(batch.batch['advantages'].shape[0], dtype=torch.bool, device=batch.batch['advantages'].device)

    response_mask = batch.batch["response_mask"]
    old_log_probs = batch.batch["old_log_probs"]
    advantages = batch.batch["advantages"]
    batch_size = advantages.shape[0]
    response_length = advantages.shape[1]
    attention_mask = batch.batch["attention_mask"]
    attention_mask_response = attention_mask[:, -response_length:]

    turn_boundaries = compute_turn_boundaries(response_mask)

    s_star = compute_s_star(old_log_probs, entropys, response_mask, turn_boundaries)
    h_wm_nll = compute_h_wm(old_log_probs, response_mask, attention_mask_response, turn_boundaries)
    h_wm_entropy = compute_h_wm_entropy(entropys, response_mask, attention_mask_response, turn_boundaries)
    h_action = compute_h_action(entropys, response_mask, turn_boundaries)
    pi_per_turn = compute_pi_per_turn(old_log_probs, response_mask, turn_boundaries)

    ref_entropy = batch.batch.get("ref_entropy", None)
    if ref_entropy is not None:
        h_wm_ref = compute_h_wm_entropy(ref_entropy, response_mask, attention_mask_response, turn_boundaries)
    else:
        h_wm_ref = None

    all_s = [s.item() for turns in s_star for s in turns]
    
    use_entropy = bool(wmc_erc_config.get("wmloss_add_use_entropy", False))
    if use_entropy:
        target_h = h_wm_entropy
    else:
        target_h = h_wm_nll

    all_h = [h.item() for turns in target_h for h in turns]

    # ===== Log stats for both H_wm (entropy) and L_wm (NLL) regardless of which is target =====
    all_h_entropy_list = [h.item() for turns in h_wm_entropy for h in turns]
    all_h_nll_list = [h.item() for turns in h_wm_nll for h in turns]

    def _stats(values):
        if not values:
            return 0.0, 0.0, 0.0
        t = torch.tensor(values, device=advantages.device, dtype=torch.float32)
        mean_t = t.mean()
        std_t = t.std(correction=0) if len(values) > 1 else torch.tensor(0.0, device=advantages.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mean_t, op=dist.ReduceOp.AVG)
            dist.all_reduce(std_t, op=dist.ReduceOp.AVG)
        mean = mean_t.item()
        std = std_t.item()
        return mean, std, std ** 2

    h_wm_entropy_mean, h_wm_entropy_std, h_wm_entropy_var = _stats(all_h_entropy_list)
    h_wm_loss_mean, h_wm_loss_std, h_wm_loss_var = _stats(all_h_nll_list)

    if not all_s:
        return batch, {}

    # Sync stats across processes
    all_s_tensor = torch.tensor(all_s, device=advantages.device, dtype=torch.float32)
    all_h_tensor = torch.tensor(all_h, device=advantages.device, dtype=torch.float32)

    batch_s_bar_t = all_s_tensor.mean()
    batch_s_std_t = all_s_tensor.std(correction=0) if len(all_s) > 1 else torch.tensor(0.0, device=advantages.device)
    batch_h_bar_t = all_h_tensor.mean()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(batch_s_bar_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_s_std_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_h_bar_t, op=dist.ReduceOp.AVG)

    batch_s_bar = batch_s_bar_t.item()
    batch_s_std = batch_s_std_t.item() + 1e-8
    batch_h_bar = batch_h_bar_t.item() + 1e-8

    momentum = wmc_erc_config.get("momentum", 0.9) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "momentum", 0.9)
    if 's_bar' not in running_stats:
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
        # Additive curiosity bonus: A' = A + coef * (factor - mean(factor))
        # factor = clip(0.5, max(0, H_wm - H_wm_mean))
        
        # Linear decay for coef
        wmloss_add_coef_start = float(wmc_erc_config.get("wmloss_add_coef", 0.1))
        wmloss_add_coef_end = float(wmc_erc_config.get("wmloss_add_coef_end", wmloss_add_coef_start))
        wmloss_add_horizon = int(wmc_erc_config.get("wmloss_add_horizon", 1))
        
        if wmloss_add_horizon > 0:
            alpha_decay = min(current_step / wmloss_add_horizon, 1.0)
            wmloss_add_coef = wmloss_add_coef_start + alpha_decay * (wmloss_add_coef_end - wmloss_add_coef_start)
        else:
            wmloss_add_coef = wmloss_add_coef_start

        wmloss_add_use_grouped = bool(wmc_erc_config.get("wmloss_add_use_grouped", False))
        wmloss_add_use_ref_baseline = bool(wmc_erc_config.get("wmloss_add_use_ref_baseline", False))
        has_group_id = 'group_id' in batch.batch.keys()

        all_offsets = []
        add_metrics = {}

        if wmloss_add_use_ref_baseline and h_wm_ref is not None:
            # Formula: offset_t = alpha * 1[failed] * clamp(H_wm - H_ref, 0, 0.5)
            # No recentering as per request
            for i in range(batch_size):
                if not traj_failed[i]:
                    continue
                for t in range(len(target_h[i])):
                    h_val = target_h[i][t]
                    h_ref_val = h_wm_ref[i][t]
                    
                    factor_t = torch.clamp(h_val - h_ref_val, min=0.0, max=0.5)
                    offset_t = wmloss_add_coef * factor_t
                    
                    start, end = turn_boundaries[i][t]
                    advantages[i, start:end] += offset_t
                    all_offsets.append(offset_t.item())
            
            # Metrics for Ref Baseline
            all_h_ref = [h.item() for turns in h_wm_ref for h in turns]
            if all_h_ref:
                add_metrics.update({
                    "wmc_erc/h_wm_ref_mean": float(np.mean(all_h_ref)),
                    "wmc_erc/h_wm_ref_std": float(np.std(all_h_ref)),
                })

        elif wmloss_add_use_grouped and has_group_id:
            gids = batch.batch['group_id'].long()
            g_max_local = gids.max() if gids.numel() > 0 else torch.tensor(-1, device=gids.device, dtype=torch.long)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(g_max_local, op=dist.ReduceOp.MAX)
            G = int(g_max_local.item()) + 1
            
            flat_h = []
            flat_gids = []
            for i in range(batch_size):
                if not traj_failed[i]:
                    continue
                for h_val in target_h[i]:
                    flat_h.append(h_val)
                    flat_gids.append(gids[i])
            
            if flat_h:
                flat_h_tensor = torch.stack(flat_h)
                flat_gids_tensor = torch.stack(flat_gids)
                
                sum_per_g = torch.zeros(G, dtype=flat_h_tensor.dtype, device=flat_h_tensor.device)
                cnt_per_g = torch.zeros(G, dtype=flat_h_tensor.dtype, device=flat_h_tensor.device)
                sum_per_g.scatter_add_(0, flat_gids_tensor, flat_h_tensor)
                cnt_per_g.scatter_add_(0, flat_gids_tensor, torch.ones_like(flat_h_tensor))
                
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(sum_per_g, op=dist.ReduceOp.SUM)
                    dist.all_reduce(cnt_per_g, op=dist.ReduceOp.SUM)
                
                mu_per_g = sum_per_g / cnt_per_g.clamp(min=1.0)
                
                # Compute factors
                factor_per_sample = []
                all_factors = []
                for i in range(batch_size):
                    sample_factors = []
                    if not traj_failed[i]:
                        factor_per_sample.append(sample_factors)
                        continue
                    gid = gids[i].item()
                    mu_h_g = mu_per_g[gid]
                    for h_val in target_h[i]:
                        factor_t = torch.clamp(h_val - mu_h_g, min=0.0, max=0.5)
                        sample_factors.append(factor_t)
                        all_factors.append(factor_t)
                    factor_per_sample.append(sample_factors)
                
                # Compute grouped mean of factors
                if all_factors:
                    all_factors_tensor = torch.stack(all_factors)
                    sum_fac_per_g = torch.zeros(G, dtype=all_factors_tensor.dtype, device=all_factors_tensor.device)
                    sum_fac_per_g.scatter_add_(0, flat_gids_tensor, all_factors_tensor)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(sum_fac_per_g, op=dist.ReduceOp.SUM)
                    mu_fac_per_g = sum_fac_per_g / cnt_per_g.clamp(min=1.0)
                    
                    # Apply grouped offset
                    for i in range(batch_size):
                        if not traj_failed[i]:
                            continue
                        gid = gids[i].item()
                        mu_fac_g = mu_fac_per_g[gid]
                        for t in range(len(factor_per_sample[i])):
                            offset_t = wmloss_add_coef * (factor_per_sample[i][t] - mu_fac_g)
                            start, end = turn_boundaries[i][t]
                            advantages[i, start:end] += offset_t
                            all_offsets.append(offset_t.item())
                
                add_metrics.update({
                    "wmc_erc/wmloss_mu_per_group_mean": mu_per_g[cnt_per_g > 0].mean().item() if (cnt_per_g > 0).any() else 0.0,
                    "wmc_erc/wmloss_mu_per_group_std": mu_per_g[cnt_per_g > 0].std(unbiased=False).item() if (cnt_per_g > 0).sum() > 1 else 0.0,
                })
        else:
            # Fallback to Global or EMA
            # General formula:
            #   offset_t = α * weight * (H_wm[t] - μ_H)
            # weight = (1 - π_t) if wmloss_add_use_pi_weight else 1.0
            # μ_H    = batch_h_bar if wmloss_add_baseline == "batch_mean" else running_stats["h_bar"] (EMA)
            # H_wm[t] = target_h, which is h_wm_entropy or h_wm_nll based on wmloss_add_use_entropy
            use_pi_weight = bool(wmc_erc_config.get("wmloss_add_use_pi_weight", True))
            baseline_type = str(wmc_erc_config.get("wmloss_add_baseline", "ema")).lower()
            if baseline_type == "batch_mean":
                mu_h = batch_h_bar
            else:
                mu_h = running_stats["h_bar"]

            for i in range(batch_size):
                if not traj_failed[i]:
                    continue
                for t in range(len(target_h[i])):
                    h_val = target_h[i][t]
                    pi_t = pi_per_turn[i][t]

                    weight = (1.0 - pi_t) if use_pi_weight else 1.0
                    offset_t = wmloss_add_coef * weight * (h_val - mu_h)

                    start, end = turn_boundaries[i][t]
                    advantages[i, start:end] += offset_t
                    all_offsets.append(offset_t.item())

        if all_offsets:
            batch.batch["advantages"] = advantages
            add_metrics.update({
                "wmc_erc/wmloss_offset_mean": float(np.mean(all_offsets)),
                "wmc_erc/wmloss_offset_std": float(np.std(all_offsets)),
                "wmc_erc/wmloss_offset_max": float(np.max(all_offsets)),
                "wmc_erc/wmloss_offset_min": float(np.min(all_offsets)),
                "wmc_erc/wmloss_coef": float(wmloss_add_coef),
            })
        else:
            add_metrics = {}
        
        # Track number of failed trajectories
        add_metrics["wmc_erc/num_failed_trajs"] = int(traj_failed.sum().item())
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
        # World model entropy stats (across all turns in batch)
        "wmc_erc/h_wm_entropy_mean": float(h_wm_entropy_mean),
        "wmc_erc/h_wm_entropy_std": float(h_wm_entropy_std),
        "wmc_erc/h_wm_entropy_var": float(h_wm_entropy_var),
        # World model loss (NLL) stats (across all turns in batch)
        "wmc_erc/wm_loss_mean": float(h_wm_loss_mean),
        "wmc_erc/wm_loss_std": float(h_wm_loss_std),
        "wmc_erc/wm_loss_var": float(h_wm_loss_var),
    }
    metrics.update(add_metrics)

    return batch, metrics
