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
from typing import Dict, List, Optional, Tuple

DEFAULT_PLAN_PROMPT = (
    "Plan ahead: list the next {k} actions you will take to make progress on the "
    "task, starting with the action you take right now, one action per line."
)

# Subgoal target prompt (plan_forecast_target='subgoal'): predict the next K
# sub-goals you will actually COMPLETE (hindsight-confirmed milestones).
DEFAULT_SUBGOAL_PROMPT = (
    "List the next {k} sub-goals you will actually accomplish from here — the "
    "milestones you will complete — one per line."
)

# ALFWORLD action grammar + object grounding, appended to the inline instructions.
# GPU env-rollout probe (base 7B, 8 episodes): adding this brought plan-mode's
# 'Nothing happens' rate down to ~52% == the no-plan floor (52%), with plan 100%
# — i.e. the cold-start tax of invalid actions disappears, so NO warmup is needed.
# NOTE: alfworld-specific verbs; for other envs this block should be swapped.
_ALFWORLD_ACTION_GRAMMAR = (
    "\n\nValid actions are EXACTLY of these forms (use these verbs verbatim, fill the "
    "slots with objects/receptacles you actually see; do NOT invent verbs like 'look "
    "at', 'place', or 'pick up'):\n"
    "go to <receptacle>\ntake <object> from <receptacle>\nput <object> in/on <receptacle>\n"
    "open <receptacle>\nclose <receptacle>\nuse <object>\nheat <object> with <receptacle>\n"
    "cool <object> with <receptacle>\nclean <object> with <receptacle>\n"
    "examine <object-or-receptacle>\nlook\ninventory\n"
    "Only interact with objects and receptacles that ACTUALLY appear in the observations; "
    "never invent objects, receptacles, or numbers. The full list of available actions is "
    "given in the FIRST observation. If unsure what is available, use 'look' or 'inventory'."
)

# Block-1 standing instruction: appended to the task instruction so the model,
# EACH turn, first writes a K-action Plan (which precedes the Action and is part of
# the generated turn, so it eats PG), then Thought, then Action. No new parsing —
# the env still reads the line after "Action:".
#
# Format chosen by GPU probe (probe_inline_plan.py, Qwen2.5-3B alfworld step75):
# a tagged "Plan:" section with a bare numbered skeleton "1. ...\n2. ..." hits
# 100% plan-compliance + 100% action-extractable, and is env-agnostic. The old
# soft "in your THOUGHT, lay out..." phrasing got 0% (model ignored it); verbose
# "<placeholder>" examples also got 0%; concrete few-shot examples are env-specific.
def inline_plan_reminder(k: int = 3) -> str:
    """Short per-turn reminder appended to EVERY observation when block-1 inline
    plan is on (a one-time standing instruction decays over turns). Kept terse to
    limit per-turn token bloat; the full format lives in the standing instruction.
    Order: Thought -> Plan -> Action."""
    return (f"\n\n[Reminder] Respond with 'Thought:' then 'Plan:' (a numbered list of "
            f"your next {k} actions, step 1 = the action you take now), then 'Action:' "
            f"(= step 1 of the Plan).")


def think_reminder() -> str:
    """Per-turn THINK reminder (a lighter alternative to the inline-plan reminder,
    mutually exclusive with it): only nudges the model to reason in a 'Thought:'
    line before the 'Action:', without forcing a forward Plan. Keeps the model
    reflective/reactive (preserves recovery actions like 'help' and obs-dependent
    choices) without the plan's forward-commitment downsides."""
    return ("\n\n[Reminder] Think before you act: first write a brief 'Thought:' "
            "reasoning about the current observation and what to do next, then give "
            "your 'Action:'.")


def inline_plan_instruction(k: int = 3) -> str:
    """Block-1 standing instruction (actions style), used as a FULL REPLACEMENT of
    the env's instruction (not appended). GPU multi-turn probe: appending decays to
    ~15% after turn 1 (the env's THOUGHT/ACTION framing wins); REPLACING with this
    reference-style instruction sustains ~95% plan-compliance across turns. Re-think
    a fresh full plan each turn; Plan -> Action (no Thought)."""
    skeleton = "\n".join(f"{i}. ..." for i in range(1, max(1, k) + 1))
    return (
        "Interact with a household to solve a task. You are an intelligent agent in a "
        "household environment; act to complete the goal. At the start you are given the "
        "environment description, your goal, and the AVAILABLE ACTIONS; each turn the "
        "environment gives feedback.\n\n"
        "On EVERY turn, output in EXACTLY this format:\n"
        f"Plan:\n{skeleton}\nAction:\nyour next action\n\n"
        f"(1) Plan: re-think from scratch a short plan of your next {k} actions from your "
        "CURRENT situation to the goal (independent each turn).\n"
        "(2) Action: your next action.\n"
        "Reminder:\n"
        "1. The Action MUST be chosen from the given AVAILABLE ACTIONS, written EXACTLY as "
        "listed. Any action other than the provided available actions is ILLEGAL and does "
        "nothing.\n"
        "2. If the environment says 'Nothing happens', the previous action was invalid — "
        "revise your plan and try a DIFFERENT available action.\n"
        "3. Output the Plan and the Action every single turn; never skip the Plan."
        + _ALFWORLD_ACTION_GRAMMAR
    )


def todo_plan_instruction(k: int = 3) -> str:
    """Block-1 standing instruction (TODO style, pairs with plan_forecast_target=
    'subgoal'), used as a FULL REPLACEMENT of the env's instruction. The Plan is a
    running TODO list; completed sub-goals are marked '(done)', which we
    hindsight-relabel as the achieved-subgoal targets. REPLACE-mode sustains 100%
    plan-compliance across turns in the probe (append-mode collapses)."""
    return (
        "Interact with a household to solve a task. You are an intelligent agent in a "
        "household environment; act to complete the goal. At the start you are given the "
        "environment description, your goal, and the AVAILABLE ACTIONS; each turn the "
        "environment gives feedback.\n\n"
        "On EVERY turn, output in EXACTLY this format:\n"
        "Plan:\n1. <sub-goal> (done)\n2. <sub-goal>\n3. <sub-goal>\nAction:\nyour next action\n\n"
        "(1) Plan: a TODO list of the sub-goals needed to reach the goal; append ' (done)' "
        "to completed sub-goals and keep/revise the rest, carrying the same sub-goals "
        "across turns.\n"
        "(2) Action: your next action.\n"
        "Reminder:\n"
        "1. The Action MUST be chosen from the given AVAILABLE ACTIONS, written EXACTLY as "
        "listed. Any action other than the provided available actions is ILLEGAL and does "
        "nothing.\n"
        "2. If the environment says 'Nothing happens', the previous action was invalid — "
        "revise your plan and try a DIFFERENT available action.\n"
        "3. Output the Plan and the Action every single turn; never skip the Plan."
        + _ALFWORLD_ACTION_GRAMMAR
    )


def todo_plan_reminder(k: int = 3) -> str:
    """Short per-turn reminder for the TODO-list plan."""
    return ("\n\n[Reminder] Maintain your Plan as a TODO list of sub-goals: append "
            "' (done)' to completed ones and keep/revise the rest. Format: 'Thought:', "
            "then 'Plan:' (numbered sub-goals, '(done)' on finished ones), then 'Action:'.")


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


_DONE_RE = re.compile(r"\(done\)|\[done\]|\[x\]|✓|✔", re.I)


def parse_plan_subgoals(turn_text: str) -> List[Tuple[str, bool]]:
    """From one assistant turn, return [(subgoal_text, is_done), ...] for the
    numbered lines in the 'Plan:' section (between 'Plan:' and 'Thought:'/'Action:').
    Done markers: '(done)', '[done]', '[x]', '✓'. The marker is stripped from text."""
    lt = turn_text or ""
    low = lt.lower()
    p = low.find("plan:")
    if p < 0:
        return []
    end = len(lt)
    for marker in ("thought:", "action:"):
        m = low.find(marker, p + 5)
        if m >= 0:
            end = min(end, m)
    seg = lt[p:end]
    out: List[Tuple[str, bool]] = []
    for line in seg.splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+)", line)
        if not m:
            continue
        item = m.group(1).strip()
        done = bool(_DONE_RE.search(item))
        clean = _DONE_RE.sub("", item).strip().strip("-—:").strip()
        if clean:
            out.append((clean, done))
    return out


def _norm_sub(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def achieved_subgoals(messages) -> List[Tuple[str, int]]:
    """Ordered [(subgoal_text, step_idx), ...] of sub-goals the model MARKED done,
    each recorded at the FIRST action-turn (step_idx) where it appears with '(done)'
    (its text there = the hindsight-confirmed version). Deduped by normalized text.
    A failed trajectory still yields its achieved sub-goals -> usable signal."""
    convo = _to_chat_list(messages)
    action_idxs = _action_turn_indices(convo)
    seen, out = set(), []
    for n, ai in enumerate(action_idxs):
        for text, done in parse_plan_subgoals(convo[ai].get('content', '') or ''):
            if not done:
                continue
            key = _norm_sub(text)
            if key in seen:
                continue
            seen.add(key)
            out.append((text, n))
    return out


def build_plan_targets(messages, k: int = 3) -> List[Dict[str, object]]:
    """For each action turn t, return per-step targets.

    {'prefix_end': idx, 'actions': [a_t..a_{t+K-1}], 'subgoals': [..]}

    ``prefix_end`` is the conversation index of the obs the action responds to
    (= action_turn_index - 1); the SFT prefix is convo[:prefix_end+1].
    ``actions`` are the realized next-K bare action commands (current included).
    ``subgoals`` are the next-K hindsight-confirmed achieved sub-goals (TODO items
    that got marked (done) at or after this step). Steps with no future action are
    skipped.
    """
    convo = _to_chat_list(messages)
    action_idxs = _action_turn_indices(convo)
    actions_seq = [extract_action(convo[ai]['content']) for ai in action_idxs]
    ach = achieved_subgoals(messages)   # [(text, step_idx)] hindsight-confirmed milestones

    out: List[Dict[str, object]] = []
    for n, ai in enumerate(action_idxs):
        fut = [a for a in actions_seq[n:n + k] if a]
        if not fut:
            continue
        # next-K sub-goals that actually complete at or after this step (hindsight)
        fut_sub = [text for (text, sn) in ach if sn >= n][:k]
        out.append({'prefix_end': ai - 1, 'actions': fut, 'subgoals': fut_sub})
    return out


def plan_block(actions: List[str]) -> str:
    """Render realized actions as an inline Plan block, matching the block-1
    instruction format: 'Plan:\\n1. a\\n2. b\\n...'."""
    return "Plan:\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1))


def build_plan_forecast_samples(
    messages,
    tokenizer,
    k: int = 3,
    target: str = "action",
    seq: str = "separate",
    max_length: int = 4096,
    min_target_tokens: int = 1,
) -> List[Dict[str, "object"]]:
    """Per-step teacher-forced forecast-SFT samples for one trajectory.

    Two orthogonal axes (the only ones after cleanup):

    target ∈ {action, subgoal}
      action  : predict the realized next-K bare action commands (block2-success,
                grounded — does NOT contaminate the no-plan rollout).
      subgoal : predict the next-K hindsight-confirmed achieved sub-goals (the
                TODO items that actually got marked (done) later).

    seq ∈ {separate, inline_consistent}
      separate          : block2 construction. prefix = convo[:obs_t+1] + a
                          synthetic user prompt ("list the next K ..."); target =
                          assistant(bare newline list, +EOS). Standalone — distinct
                          from the rollout turn. Use with inline plan OFF.
      inline_consistent : build the SFT sample to MATCH a real rollout turn so the
                          SFT reinforces (not corrupts) the rollout format. prefix =
                          convo[:obs_t+1] (the obs, NO synthetic prompt); target =
                          a full assistant turn 'Plan:\\n1. ...\\nAction:\\n<a_t>'
                          (Plan block of the realized items + the grounded current
                          action). Use with inline plan ON.

    Loss mask covers only the target tokens. Returns dicts with torch tensors.
    """
    import torch
    convo = _to_chat_list(messages)
    sep_prompt = (DEFAULT_SUBGOAL_PROMPT if target == "subgoal"
                  else DEFAULT_PLAN_PROMPT).format(k=k)

    samples: List[Dict[str, object]] = []
    for tgt in build_plan_targets(messages, k=k):
        items = (tgt.get('subgoals') if target == "subgoal"
                 else tgt.get('actions')) or []
        if not items:
            continue
        if seq == "inline_consistent":
            # SFT sample == a real rollout turn: obs -> assistant(Plan + Action).
            # Plan = realized next-K items; Action = the grounded current action.
            prefix = list(convo[:tgt['prefix_end'] + 1])
            action = (tgt.get('actions') or [""])[0]
            content = f"{plan_block(items)}\nAction:\n{action}"
            target_msgs = [{'role': 'assistant', 'content': content}]
        else:  # separate
            prefix = list(convo[:tgt['prefix_end'] + 1])
            prefix.append({'role': 'user', 'content': sep_prompt})
            target_msgs = [{'role': 'assistant', 'content': "\n".join(items)}]
        try:
            prefix_text = tokenizer.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                prefix + target_msgs, tokenize=False, add_generation_prompt=False)
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
    target: str = "action",
    seq: str = "separate",
    max_length: int = 4096,
    max_samples_per_trajectory: Optional[int] = None,
):
    """Padded forecast-SFT batch over trajectories, with win/all gating.

    gate='wins' keeps only trajectories whose reward > success_threshold (needs
    ``rewards`` aligned to ``messages_list``); gate='all' keeps everything.
    target ∈ {action, subgoal}; seq ∈ {separate, inline_consistent} — see
    build_plan_forecast_samples. Reuses collate_world_model_samples for padding.
    """
    from verl.agent_trainer.ppo.world_model_loss import collate_world_model_samples

    all_samples: List[Dict[str, object]] = []
    n_traj_used = 0
    # done/sub-goal monitoring (only meaningful for target='subgoal')
    n_traj_considered = 0
    n_achieved_total = 0
    n_traj_with_done = 0
    for i, messages in enumerate(messages_list):
        if messages is None:
            continue
        if gate == "wins":
            r = rewards[i] if (rewards is not None and i < len(rewards)) else 0.0
            if not (r is not None and float(r) > success_threshold):
                continue
        n_traj_considered += 1
        if target == "subgoal":
            ach = achieved_subgoals(messages)
            n_achieved_total += len(ach)
            n_traj_with_done += 1 if ach else 0
        traj_samples = build_plan_forecast_samples(
            messages=messages, tokenizer=tokenizer, k=k,
            target=target, seq=seq, max_length=max_length)
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
    _cons = max(1, n_traj_considered)
    meta = {"plan_forecast/n_samples": float(len(all_samples)),
            "plan_forecast/n_traj_used": float(n_traj_used),
            "plan_forecast/n_traj_considered": float(n_traj_considered),
            "plan_forecast/gate_wins": 1.0 if gate == "wins" else 0.0,
            "plan_forecast/seq_inline_consistent": 1.0 if seq == "inline_consistent" else 0.0,
            "plan_forecast/target_subgoal": 1.0 if target == "subgoal" else 0.0,
            "plan_forecast/k": float(k)}
    if target == "subgoal":
        # done-marking health: is the LLM actually checking sub-goals off?
        meta.update({
            "plan_forecast/n_achieved_subgoals": float(n_achieved_total),
            "plan_forecast/achieved_per_traj": float(n_achieved_total) / _cons,
            "plan_forecast/frac_traj_with_done": float(n_traj_with_done) / _cons,
            "plan_forecast/samples_per_traj": float(len(all_samples)) / _cons,
        })
    return batch, meta
