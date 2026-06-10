# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.agent_trainer.ppo.world_model_loss import (
    compute_world_model_loss,
    compute_world_model_sft_loss_from_logits,
    compute_observation_mask,
)
from verl.workers.agent_actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class ValueHead(nn.Module):
    """Small MLP value head attached to the actor backbone.

    Linear(hidden, hidden//mid_ratio) -> Tanh -> Linear(hidden//mid_ratio, 1)
    -> Sigmoid.

    The final sigmoid bounds V ∈ (0, 1) so it lives on the same scale as the
    outcome reward R ∈ {0, 1} (alfworld/sciworld). This eliminates the
    blow-ups we saw with an unbounded head (e.g. V_mean=1.012 mid-training)
    that injected huge swings into the per-turn redistribution
    A_t = Â − β·(V̂_t − mean_t V̂_t).

    fc2 is zero-initialized so logits start at 0, V ≡ sigmoid(0) = 0.5 at
    every state. The redistribution term is thus exactly zero at init
    (delta_t = V_t − mean_t V_t = 0.5 − 0.5 = 0), preserving the soft-start
    semantics from the previous unbounded design.
    """

    def __init__(self, hidden_size: int, mid_ratio: int = 4):
        super().__init__()
        mid = max(1, hidden_size // max(1, mid_ratio))
        self.fc1 = nn.Linear(hidden_size, mid)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(mid, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(self.act(self.fc1(hidden)))).squeeze(-1)


def compute_turn_ids(response_mask: torch.Tensor) -> torch.Tensor:
    """Identify distinct assistant-token turns in the response.
    
    Returns a tensor of the same shape as response_mask, where each assistant
    turn is labeled with a unique ID (0, 1, ...), and non-assistant tokens
    are ignored.
    """
    turn_ids = torch.zeros_like(response_mask, dtype=torch.long)
    for i in range(response_mask.shape[0]):
        mask = response_mask[i].bool()
        if not mask.any():
            continue
        
        # turn_id increments on each 0->1 transition in response_mask
        turn_id = 0
        in_turn = False
        for j in range(len(mask)):
            if mask[j]:
                if not in_turn:
                    if j > 0 and any(mask[:j]): # Increment only after the first turn
                        turn_id += 1
                    in_turn = True
                turn_ids[i, j] = turn_id
            else:
                in_turn = False
    return turn_ids


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        model_config=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

        # Optional value head (Linear->Tanh->Linear) used by the value-baseline
        # branch. Gated entirely behind `use_value_baseline`; when disabled the
        # attribute stays None and every code path is identical to the GRPO
        # default.
        self.use_value_baseline = bool(self.config.get('use_value_baseline', False))
        self.value_loss_coef = float(self.config.get('value_loss_coef', 0.5))
        self.value_baseline_beta = float(self.config.get('value_baseline_beta', 1.0))
        self.value_head_mid_ratio = int(self.config.get('value_head_mid_ratio', 4))
        # Independent lr for the value head; defaults to 1e-3 (typical PPO
        # critic lr), 1000x the actor lr. Critical because the value head
        # starts from a zero-initialized fc2 and would otherwise barely move
        # at the actor's 1e-6.
        self.value_head_lr = float(self.config.get('value_head_lr', 1e-3))
        self.value_head = None
        if self.use_value_baseline and actor_optimizer is not None and model_config is not None:
            hidden_size = int(getattr(model_config, 'hidden_size'))
            self.value_head = ValueHead(hidden_size=hidden_size,
                                        mid_ratio=self.value_head_mid_ratio).to(
                device=torch.cuda.current_device(), dtype=torch.float32)
            # Separate param group with its own lr — the actor lr (≈1e-6) is
            # far too small for a fresh small MLP starting from zero-init.
            self.actor_optimizer.add_param_group({
                'params': list(self.value_head.parameters()),
                'lr': self.value_head_lr,
            })
            print(f'[value-baseline] enabled; hidden_size={hidden_size}, '
                  f'mid_ratio={self.value_head_mid_ratio}, lr={self.value_head_lr:g}, '
                  f'beta={self.value_baseline_beta}, vloss_coef={self.value_loss_coef}')

        # Optional Hindsight Credit Assignment (HCAPO, arXiv:2603.08754).
        # Training-free "Generative Verification": the SAME frozen policy is
        # re-prompted with the realized outcome (s_final) injected as a prefix
        # in the context, and we recompute its log-prob of the action tokens.
        # No separate head, no SFT — this is the paper's core design and
        # avoids the moving-target h-fit problem of classical HCA entirely.
        # Hard mutex with value baseline.
        self.use_hindsight_hca = bool(self.config.get('use_hindsight_hca', False))
        assert not (self.use_value_baseline and self.use_hindsight_hca), \
            "USE_VALUE_BASELINE and USE_HINDSIGHT_HCA are mutually exclusive."
        # R > threshold ⇒ outcome label z=1 (success), used to pick the prefix.
        self.hca_z_threshold = float(self.config.get('hca_z_threshold', 0.5))
        # Sharpening temperature T_temp in π_hind = exp(mean_log_p / T_temp).
        self.hca_temp = float(self.config.get('hca_temp', 5.0))
        if self.use_hindsight_hca:
            print(f'[hindsight-hca] training-free generative verification; '
                  f'z_threshold={self.hca_z_threshold}, temp={self.hca_temp}')

    def _forward_micro_batch(self, micro_batch, temperature, return_hidden_states: bool = False):
        """
        Returns:
            entropy: (bs, response_len)
            log_probs: (bs, response_len)
            full_hidden (optional, when return_hidden_states=True):
                (bs, seqlen, hidden) — last-layer hidden states over the FULL
                input sequence (prompt + response), needed for state-value
                queries at positions in the prompt region (turn 0).
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           output_hidden_states=return_hidden_states,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # also pull the last-layer hidden state if requested
                last_hidden_rmpad = None
                if return_hidden_states:
                    last_hidden_rmpad = output.hidden_states[-1].squeeze(0)  # (total_nnz_sp, hidden)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    if last_hidden_rmpad is not None:
                        last_hidden_rmpad = gather_outpus_and_unpad(last_hidden_rmpad,
                                                                    gather_dim=0,
                                                                    unpad_dim=0,
                                                                    padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

                if return_hidden_states:
                    full_hidden = pad_input(hidden_states=last_hidden_rmpad,
                                            indices=indices,
                                            batch=batch_size,
                                            seqlen=seqlen)  # (bsz, seqlen, hidden)
                    return entropy, log_probs, full_hidden

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           output_hidden_states=return_hidden_states,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                if return_hidden_states:
                    full_hidden = output.hidden_states[-1]  # (bsz, seqlen, hidden)
                    return entropy, log_probs, full_hidden

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: log_prob tensor and,
            optionally, entropy tensor over response tokens.
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        return_entropy = data.meta_info.get('return_entropy', False)

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            if return_entropy:
                entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if return_entropy else None

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if return_entropy:
                entropys = entropys[revert_indices]

        if return_entropy:
            return log_probs, entropys
        return log_probs

    def _state_positions_per_sample(self, response_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """For each sample build the list of state positions (one per turn).

        A state position is the input_ids index ``prompt_len + start - 1`` where
        ``start`` is the first action-token index (in response coords) of a
        turn. ``prompt_len`` is inferred as ``seqlen - response_len``.

        Returns
        -------
        positions : (B, max_turns) long
            Padded with 0; meaningful only where mask is True.
        mask : (B, max_turns) bool
            True for valid turn slots.
        max_turns : int
        """
        # Use compute_turn_boundaries from wmc_erc to stay consistent with
        # the rest of the pipeline.
        from verl.agent_trainer.ppo.wmc_erc import compute_turn_boundaries
        boundaries_per_sample = compute_turn_boundaries(response_mask.cpu())
        B = response_mask.size(0)
        max_turns = max((len(b) for b in boundaries_per_sample), default=0)
        positions = torch.zeros(B, max(1, max_turns), dtype=torch.long)
        mask = torch.zeros(B, max(1, max_turns), dtype=torch.bool)
        for i, boundaries in enumerate(boundaries_per_sample):
            for t, (start, _) in enumerate(boundaries):
                # In input_ids coords; ``prompt_len`` is implicit via offset.
                # We store the response-coord ``start`` here and adjust to
                # full-seq coord at lookup time using the caller's offset.
                positions[i, t] = start
                mask[i, t] = True
        return positions, mask, max(1, max_turns)

    def compute_state_values(self, data: DataProto) -> torch.Tensor:
        """Compute per-turn state values V(s_t) for every sample.

        Forwards once with output_hidden_states=True (no grad), gathers the
        last-layer hidden state at the position right BEFORE the first action
        token of each turn, applies the value head, and returns
        ``state_values`` shape (B, max_turns) padded with 0 at invalid slots
        and ``state_values_mask`` shape (B, max_turns) bool.
        """
        assert self.value_head is not None, 'compute_state_values requires value_head'
        self.actor_module.eval()
        self.value_head.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'response_mask']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        v_per_micro = []
        mask_per_micro = []
        for micro_batch in micro_batches:
            response_length = micro_batch['responses'].size(-1)
            seqlen = micro_batch['input_ids'].size(1)
            prompt_offset = seqlen - response_length
            response_mask = micro_batch['response_mask']
            positions, mask, max_turns = self._state_positions_per_sample(response_mask)

            with torch.no_grad():
                _, _, full_hidden = self._forward_micro_batch(micro_batch, temperature=temperature,
                                                              return_hidden_states=True)
                # full_hidden: (b, seqlen, hidden)
                positions = positions.to(full_hidden.device)
                mask = mask.to(full_hidden.device)
                # state position in full-seq coords: prompt_offset + start - 1
                full_positions = (prompt_offset + positions - 1).clamp(min=0)
                idx = full_positions.unsqueeze(-1).expand(-1, -1, full_hidden.size(-1))
                gathered = full_hidden.gather(1, idx)  # (b, max_turns, hidden)
                values = self.value_head(gathered.to(self.value_head.fc1.weight.dtype))
                values = values * mask.to(values.dtype)

            v_per_micro.append(values.float().cpu())
            mask_per_micro.append(mask.cpu())

        # Pad each micro-batch's per-turn dim to a common max_turns and stack.
        max_turns_global = max(v.size(1) for v in v_per_micro) if v_per_micro else 1
        def _pad_to(v, T):
            if v.size(1) >= T:
                return v[:, :T]
            pad = torch.zeros(v.size(0), T - v.size(1), dtype=v.dtype)
            return torch.cat([v, pad], dim=1)

        values_out = torch.cat([_pad_to(v, max_turns_global) for v in v_per_micro], dim=0)
        mask_out = torch.cat([_pad_to(m.to(torch.float32), max_turns_global) for m in mask_per_micro], dim=0).bool()

        if use_dynamic_bsz:
            flat_indices = list(itertools.chain.from_iterable(indices))
            assert len(flat_indices) == values_out.size(0)
            revert_indices = torch.tensor(get_reverse_idx(flat_indices), dtype=torch.long)
            values_out = values_out[revert_indices]
            mask_out = mask_out[revert_indices]

        return values_out, mask_out

    # ============================================================
    # Hindsight Credit Assignment (HCA) — see feature.md
    # ============================================================

    def _get_lm_head(self):
        """Return the actor's output embedding module (a.k.a. LM head).

        Goes through ``get_output_embeddings()`` which most HF causal LMs
        expose. FSDP-wrapped modules forward this call to the wrapped model.
        """
        return self.actor_module.get_output_embeddings()

    def _outcome_z(self, traj_return: torch.Tensor) -> torch.Tensor:
        """Binary outcome label: 1 if R > threshold else 0."""
        return (traj_return > self.hca_z_threshold).long()

    def _hindsight_forward_genver(self, micro_batch, temperature, z,
                                  prefix_ids, prefix_mask):
        """Training-free Generative Verification (HCAPO §4.2).

        Re-prompt the SAME frozen policy with the realized outcome injected as
        a token prefix at the front of the context, then read off the policy's
        own log-prob of the action tokens it actually produced. No head, no
        gradient — this is pure inference with the policy weights.

        prefix_ids/prefix_mask: (num_z, K) outcome-statement token ids / mask,
            indexed by the per-sample outcome label z. Left-padded to a common
            length K so every sample in the batch grows by the same K tokens.

        Returns: h_log_probs of shape (B, response_len) — log π(a_t | s_t, z).
        """
        response_length = micro_batch['responses'].size(-1)
        input_ids = micro_batch['input_ids']
        attention_mask = micro_batch['attention_mask']
        B = input_ids.size(0)

        pre_ids = prefix_ids[z]                                  # (B, K)
        pre_mask = prefix_mask[z].to(attention_mask.dtype)       # (B, K)
        new_input_ids = torch.cat([pre_ids, input_ids], dim=1)
        new_attn = torch.cat([pre_mask, attention_mask], dim=1)
        # Recompute position_ids from the augmented attention mask so RoPE sees
        # the prefix→content order correctly regardless of original padding.
        new_pos = (new_attn.long().cumsum(dim=-1) - 1).clamp(min=0)
        new_pos = new_pos * new_attn.long()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                output = self.actor_module(
                    input_ids=new_input_ids,
                    attention_mask=new_attn,
                    position_ids=new_pos,
                    use_cache=False,
                )
            logits = output.logits / max(temperature, 1e-6)
            # Response tokens remain the last `response_length` tokens; the
            # prefix only shifts the front, so this tail-slice still selects
            # the logits that predict the action tokens.
            logits = logits[:, -response_length - 1:-1, :]
            h_log_probs = logprobs_from_logits(logits, micro_batch['responses'])
        return h_log_probs

    def compute_hindsight_log_probs(self, data: DataProto):
        """Inference: log π(a_t | s_t, s_final) at response token positions for
        every (sample, action token), under outcome-conditioned context. Used
        to form the hindsight importance ratio ρ = π_hind / π̄_hind.

        Reads the outcome-statement prefixes from data.meta_info
        ('hca_prefix_ids', 'hca_prefix_mask'), built by the worker tokenizer.

        Returns: h_log_probs tensor of shape (B, response_len) on CPU.
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        dev = torch.cuda.current_device()
        prefix_ids = data.meta_info['hca_prefix_ids'].to(dev)     # (num_z, K)
        prefix_mask = data.meta_info['hca_prefix_mask'].to(dev)   # (num_z, K)

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'traj_return']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        out_list = []
        for micro_batch in micro_batches:
            z = self._outcome_z(micro_batch['traj_return']).to(device=dev)
            h_lp = self._hindsight_forward_genver(
                micro_batch, temperature, z, prefix_ids, prefix_mask)
            out_list.append(h_lp.detach().cpu())
        h_log_probs = torch.concat(out_list, dim=0)

        if use_dynamic_bsz:
            flat_indices = list(itertools.chain.from_iterable(indices))
            assert len(flat_indices) == h_log_probs.size(0)
            revert = torch.tensor(get_reverse_idx(flat_indices), dtype=torch.long)
            h_log_probs = h_log_probs[revert]
        return h_log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        world_model_coeff = data.meta_info.get('world_model_coeff', self.config.get('world_model_coeff', 0.0))

        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'responses', 'response_mask']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if world_model_coeff > 0 and 'observation_mask' in data.batch.keys():
            select_keys.append('observation_mask')
        if self.use_value_baseline and 'traj_return' in data.batch.keys():
            select_keys.append('traj_return')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                response_mask = data['response_mask']
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                need_hidden = self.use_value_baseline and self.value_head is not None
                if need_hidden:
                    entropy, log_prob, full_hidden = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, return_hidden_states=True)
                else:
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                             log_prob=log_prob,
                                                                             advantages=advantages,
                                                                             eos_mask=response_mask,
                                                                             cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                # Legacy in-place world-model SFT term: only fires when the
                # batch carries an explicit ``observation_mask`` (e.g. tests or
                # callers that still mask env tokens on the rollout sequence).
                # The recommended path is the separate ``update_world_model``
                # pass driven by ``ray_trainer.fit`` on a freshly re-assembled
                # chat-template batch.
                if world_model_coeff > 0:
                    explicit_observation_mask = data.get('observation_mask', None)
                    
                    if self.config.get('wm_loss_pi_dedup', False):
                        obs_mask = compute_observation_mask(
                            attention_mask=data['attention_mask'],
                            response_mask=response_mask,
                            observation_mask=explicit_observation_mask)
                        
                        if obs_mask.any().item():
                            olp = data['old_log_probs'].detach()
                            t_ids = compute_turn_ids(response_mask)
                            n_total = int(t_ids.max().item()) + 1 if t_ids.numel() else 1
                            B = response_mask.size(0)
                            dt = log_prob.dtype
                            rm_f = response_mask.to(dt)
                            
                            log_pi_sum = torch.zeros(B, n_total, device=olp.device, dtype=dt)
                            n_act = torch.zeros_like(log_pi_sum)
                            log_pi_sum.scatter_add_(1, t_ids, olp.to(dt) * rm_f)
                            n_act.scatter_add_(1, t_ids, rm_f)
                            
                            log_pi_mean = log_pi_sum / n_act.clamp(min=1.0)
                            pi_per_turn = log_pi_mean.exp().clamp(0.0, 1.0)
                            w_per_turn = (1.0 - pi_per_turn)  # ∈ [0, 1]
                            w_per_token = w_per_turn.gather(1, t_ids).to(dt)
                            
                            obs_mask_w = obs_mask.to(dt) * w_per_token
                            denom = obs_mask_w.sum().clamp(min=1e-6)
                            wm_sft_loss = -(log_prob * obs_mask_w).sum() / denom
                            metrics['actor/wm_sft_pi_dedup_w_mean'] = w_per_turn.mean().detach().item()
                        else:
                            wm_sft_loss = None
                    else:
                        wm_sft_loss, _ = compute_world_model_loss(
                            log_prob=log_prob,
                            attention_mask=data['attention_mask'],
                            response_mask=response_mask,
                            observation_mask=explicit_observation_mask,
                        )

                    if wm_sft_loss is not None:
                        policy_loss = policy_loss + world_model_coeff * wm_sft_loss
                        metrics['actor/wm_sft_loss'] = wm_sft_loss.detach().item()
                        metrics['actor/world_model_coeff'] = world_model_coeff

                value_loss_val = None
                value_pred_mean_val = None
                if need_hidden and 'traj_return' in data.keys():
                    # Per-sample state-value MSE loss: V(s_t) -> R (the
                    # trajectory final reward, broadcast to every turn).
                    # CRITICAL: detach hidden states before the value head so
                    # the value-MSE gradient does NOT flow into the actor
                    # backbone. The value head is intended to *read* the
                    # backbone's representation, not reshape it; without
                    # detach the value-loss term (numerically O(R²) ≈ 1)
                    # dominates the much smaller pg_loss (~O(0.1)) and pulls
                    # the policy away from the GRPO direction.
                    response_length = data['responses'].size(-1)
                    seqlen = data['input_ids'].size(1)
                    prompt_offset = seqlen - response_length
                    positions, sv_mask, _ = self._state_positions_per_sample(response_mask)
                    positions = positions.to(full_hidden.device)
                    sv_mask = sv_mask.to(full_hidden.device)
                    full_positions = (prompt_offset + positions - 1).clamp(min=0)
                    idx = full_positions.unsqueeze(-1).expand(-1, -1, full_hidden.size(-1))
                    gathered = full_hidden.detach().gather(1, idx)
                    v_pred = self.value_head(gathered.to(self.value_head.fc1.weight.dtype))  # (B, max_turns)
                    R = data['traj_return'].to(v_pred.dtype).unsqueeze(-1)  # (B, 1)
                    sv_mask_f = sv_mask.to(v_pred.dtype)
                    denom = sv_mask_f.sum().clamp(min=1.0)
                    value_loss = (((v_pred - R) ** 2) * sv_mask_f).sum() / denom
                    policy_loss = policy_loss + self.value_loss_coef * value_loss
                    value_loss_val = value_loss.detach().item()
                    value_pred_mean_val = (v_pred.detach() * sv_mask_f).sum().item() / denom.item()

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                if value_loss_val is not None:
                    data['actor/value_loss'] = value_loss_val
                    data['actor/value_pred_mean'] = value_pred_mean_val
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    def update_world_model(self, data: DataProto):
        """SFT update on a freshly assembled world-model batch.

        Expects ``data.batch`` with keys ``input_ids``, ``attention_mask``,
        ``position_ids`` and ``loss_mask`` (1 on env-observation tokens the
        model should learn to predict).  The resulting CE loss is scaled by
        ``self.config.world_model_coeff`` before ``.backward()``.
        """
        self.actor_module.train()

        coef = data.meta_info.get('world_model_coeff', float(self.config.get('world_model_coeff', 0.0)))
        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'loss_mask']

        batch = data.select(batch_keys=select_keys).batch

        mini_batch_size = self.config.get('world_model_mini_batch_size',
                                          self.config.ppo_mini_batch_size)
        micro_batch_size = self.config.get('world_model_micro_batch_size_per_gpu',
                                           self.config.ppo_micro_batch_size_per_gpu)

        metrics: dict = {}
        dataloader = batch.split(mini_batch_size) if mini_batch_size else [batch]

        for mini_batch in dataloader:
            micro_batches = mini_batch.split(micro_batch_size) if micro_batch_size else [mini_batch]
            gradient_accumulation = max(1, len(micro_batches))

            self.actor_optimizer.zero_grad()
            for micro in micro_batches:
                micro = micro.cuda()
                loss_mask = micro['loss_mask']
                if loss_mask.sum().item() == 0:
                    continue

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = self.actor_module(
                        input_ids=micro['input_ids'],
                        attention_mask=micro['attention_mask'],
                        position_ids=micro['position_ids'],
                        use_cache=False,
                    )
                    wm_loss = compute_world_model_sft_loss_from_logits(
                        logits=output.logits,
                        labels=micro['input_ids'],
                        loss_mask=loss_mask,
                    )

                loss = coef * wm_loss / gradient_accumulation
                loss.backward()

                append_to_dict(metrics, {
                    'actor/world_model_sft_loss': wm_loss.detach().item(),
                    'actor/world_model_coef': coef,
                    'actor/world_model_valid_tokens': loss_mask.sum().detach().item(),
                })

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {'actor/world_model_grad_norm': grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
