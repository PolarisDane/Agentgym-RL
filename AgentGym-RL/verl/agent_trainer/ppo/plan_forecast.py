"""Plan-forecast auxiliary loss: at every step t the model predicts the NEXT K
action commands it will take (the "plan"), supervised by the REALIZED future —
i.e. the actual actions a_t, a_{t+1}, ..., a_{t+K-1} taken in the rollout (the
current action is INCLUDED, so plan[0] == the action committed this turn).

This is the post-hoc, teacher-forced half of the design. It is the sibling of the
world-model SFT loss (world_model_loss.py): same chat-template re-assembly, same
collate + CE-from-logits, but the target is the agent's own future ACTION string
instead of the environment's next observation. It is a SEPARATE forward pass
(``update_plan_forecast``) and does NOT touch PG. The OTHER half — the inline
``<plan>`` that conditions the action and eats PG — lives in the rollout / env
adapter; the two share the backbone but operate on different token spans.

Leakage is intentionally ignored (per design): the realized future IS the target.

Gating: ``gate='wins'`` (default) keeps only trajectories with reward above
``success_threshold`` so we never teach the model to foresee a flailing future;
``gate='all'`` uses every trajectory (more data, but pulls plans toward bad
futures on losing rollouts — kept for ablation).

PURE logic for sample assembly (stdlib + tokenizer only, CPU-testable). The CE
loss + collate are reused from world_model_loss to avoid divergence.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

DEFAULT_PLAN_PROMPT = (
    "Plan ahead: list the next {k} actions you will take to make progress on the "
    "task, starting with the action you take right now, one action per line."
)

# Block-1 standing instruction: appended to the task instruction so the model,
# EACH turn, first writes its K-step plan inside the Thought (which already eats
# PG and precedes the Action), then the Action. No new parsing — the env still
# reads the line after "Action:". Shares K + "starting with the action you take
# right now" framing with the block-2 forecast target so the two are about the
# same object (inline plan eats PG; forecast supervises it against the realized
# future).
INLINE_PLAN_INSTRUCTION = (
    "\n\nAdditionally, in every THOUGHT, before deciding your action, first briefly "
    "lay out your plan for the next {k} actions you intend to take to make progress "
    "on the task, starting with the action you are about to take now. Then choose "
    "your ACTION as usual (the action you actually execute must be the first step of "
    "this plan)."
)


def inline_plan_instruction(k: int = 3) -> str:
    """The block-1 standing instruction text for a K-step inline plan."""
    return INLINE_PLAN_INSTRUCTION.format(k=k)


def _to_chat_list(messages) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            out.append({'role': m['role'], 'content': m['content']})
        elif hasattr(m, 'to_dict'):
            out.append(m.to_dict())
        else:  # pragma: no cover - defensive
            out.append({'role': getattr(m, 'role'), 'content': getattr(m, 'content')})
    return out


def extract_action(assistant_text: str) -> str:
    """The bare action command from an assistant turn (drops the Thought).

    Mirrors progress_credit_probe.parse_action: take the first non-empty line
    after ``Action:``; fall back to the last non-empty line for bare-action envs.
    Returns '' for empty/degenerate turns (e.g. the trailing terminal turn).
    """
    m = re.search(r"Action:\s*(.+)", assistant_text or "", re.S)
    if not m:
        lines = [l.strip() for l in (assistant_text or "").splitlines() if l.strip()]
        return lines[-1] if lines else ""
    for ln in m.group(1).strip().splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def _action_turn_indices(convo: List[Dict[str, str]]) -> List[int]:
    """Indices of assistant ACTION turns, in chronological order.

    Layout (codebase convention, shared with hca_perstep / progress_credit):
    [instr(user), ack(assistant), obs0(user), action0(assistant), obs1, action1, ...].
    The instruction+ack pair is skipped; action turns sit at conv idx 3, 5, 7, ...
    (assistant, each preceded by a user obs).
    """
    return [i for i in range(3, len(convo), 2)
            if convo[i]['role'] == 'assistant' and convo[i - 1]['role'] == 'user']


def build_plan_targets(messages, k: int = 3) -> List[Dict[str, object]]:
    """For each action turn t, return {'prefix_end': idx, 'actions': [a_t..a_{t+K-1}]}.

    ``prefix_end`` is the conversation index of the obs the action responds to
    (= action_turn_index - 1); the SFT prefix is convo[:prefix_end+1] + plan prompt.
    ``actions`` are the realized next-K bare action commands (current included),
    empty commands (terminal turn) dropped from the tail.
    """
    convo = _to_chat_list(messages)
    action_idxs = _action_turn_indices(convo)
    actions_seq = [extract_action(convo[ai]['content']) for ai in action_idxs]

    out: List[Dict[str, object]] = []
    for n, ai in enumerate(action_idxs):
        fut = [a for a in actions_seq[n:n + k] if a]
        if not fut:
            continue
        out.append({'prefix_end': ai - 1, 'actions': fut})
    return out


def build_plan_forecast_samples(
    messages,
    tokenizer,
    k: int = 3,
    plan_prompt: str = DEFAULT_PLAN_PROMPT,
    max_length: int = 4096,
    min_target_tokens: int = 1,
) -> List[Dict[str, "object"]]:
    """Per-step teacher-forced SFT samples for one trajectory.

    prefix = convo[:obs_t+1] + user(plan_prompt) ; target = assistant(next-K actions,
    one per line). Loss mask covers only the target tokens. Returns dicts with
    torch tensors input_ids/attention_mask/loss_mask (lazy torch import).
    """
    import torch
    convo = _to_chat_list(messages)
    prompt = plan_prompt.format(k=k)

    samples: List[Dict[str, object]] = []
    for tgt in build_plan_targets(messages, k=k):
        prefix = list(convo[:tgt['prefix_end'] + 1])
        prefix.append({'role': 'user', 'content': prompt})
        target_text = "\n".join(tgt['actions'])
        target = [{'role': 'assistant', 'content': target_text}]

        try:
            prefix_text = tokenizer.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                prefix + target, tokenize=False, add_generation_prompt=False)
        except Exception:  # pragma: no cover - tokenizer template missing
            continue

        prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                               return_tensors='pt')['input_ids'][0]
        full_ids = tokenizer(full_text, add_special_tokens=False,
                             return_tensors='pt')['input_ids'][0]
        if full_text.startswith(prefix_text):
            prefix_len = prefix_ids.size(0)
        else:
            common = 0
            for i in range(min(len(prefix_ids), len(full_ids))):
                if prefix_ids[i].item() != full_ids[i].item():
                    break
                common = i + 1
            prefix_len = common

        target_len = full_ids.size(0) - prefix_len
        if target_len < min_target_tokens:
            continue

        input_ids = full_ids
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[prefix_len:] = 1

        if input_ids.size(0) > max_length:
            drop = input_ids.size(0) - max_length
            input_ids = input_ids[drop:]
            attention_mask = attention_mask[drop:]
            loss_mask = loss_mask[drop:]
            if loss_mask.sum().item() < min_target_tokens:
                continue

        samples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
        })

    return samples


def build_plan_forecast_batch(
    messages_list,
    tokenizer,
    rewards: Optional[List[float]] = None,
    k: int = 3,
    gate: str = "wins",
    success_threshold: float = 0.5,
    plan_prompt: str = DEFAULT_PLAN_PROMPT,
    max_length: int = 4096,
    max_samples_per_trajectory: Optional[int] = None,
):
    """Padded SFT batch over trajectories, with win/all gating.

    gate='wins' keeps only trajectories whose reward > success_threshold (needs
    ``rewards`` aligned to ``messages_list``); gate='all' keeps everything.
    Reuses world_model_loss.collate_world_model_samples for padding.
    """
    from verl.agent_trainer.ppo.world_model_loss import collate_world_model_samples

    all_samples: List[Dict[str, object]] = []
    n_traj_used = 0
    for i, messages in enumerate(messages_list):
        if messages is None:
            continue
        if gate == "wins":
            r = rewards[i] if (rewards is not None and i < len(rewards)) else 0.0
            if not (r is not None and float(r) > success_threshold):
                continue
        traj_samples = build_plan_forecast_samples(
            messages=messages, tokenizer=tokenizer, k=k,
            plan_prompt=plan_prompt, max_length=max_length)
        if max_samples_per_trajectory is not None and len(traj_samples) > max_samples_per_trajectory:
            traj_samples = traj_samples[-max_samples_per_trajectory:]
        if traj_samples:
            n_traj_used += 1
        all_samples.extend(traj_samples)

    batch = collate_world_model_samples(
        samples=all_samples,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        max_length=max_length,
    )
    meta = {"plan_forecast/n_samples": float(len(all_samples)),
            "plan_forecast/n_traj_used": float(n_traj_used),
            "plan_forecast/gate_wins": 1.0 if gate == "wins" else 0.0}
    return batch, meta
