"""Pure-logic unit tests for plan_forecast (no model/tokenizer needed)."""
import importlib.util, os
_p = "AgentGym-RL/verl/agent_trainer/ppo/plan_forecast.py"
_spec = importlib.util.spec_from_file_location("plan_forecast", _p)
plan_forecast = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan_forecast)
extract_action = plan_forecast.extract_action
build_plan_targets = plan_forecast.build_plan_targets
_action_turn_indices = plan_forecast._action_turn_indices


def conv(*turns):
    """turns = (role, content) pairs."""
    return [{"role": r, "content": c} for r, c in turns]


def test_extract_action():
    assert extract_action("Thought: find it.\n\nAction:\ngo to desk 1") == "go to desk 1"
    assert extract_action("Action: take cellphone 4 from desk 1") == "take cellphone 4 from desk 1"
    assert extract_action("Action:\ngo to countertop 1") == "go to countertop 1"
    assert extract_action("") == ""
    assert extract_action("buy now") == "buy now"            # bare-action fallback
    print("ok extract_action")


def test_action_turn_indices_skips_ack():
    c = conv(("user", "instr"), ("assistant", "OK ack"),
             ("user", "obs0"), ("assistant", "Action: a0"),
             ("user", "obs1"), ("assistant", "Action: a1"))
    # ack at idx1 has no preceding user obs -> skipped; action turns at 3,5
    assert _action_turn_indices(c) == [3, 5]
    print("ok action_turn_indices")


def test_build_plan_targets_includes_current_and_window():
    c = conv(("user", "instr"), ("assistant", "OK"),
             ("user", "obs0"), ("assistant", "Action: go to desk 1"),
             ("user", "obs1"), ("assistant", "Action: take cd 1"),
             ("user", "obs2"), ("assistant", "Action: use lamp 1"),
             ("user", "obs3"), ("assistant", ""))   # terminal empty turn
    t = build_plan_targets(c, k=3)
    # step0 -> [a0,a1,a2] (current included), prefix_end = obs0 idx = 2
    assert t[0]["prefix_end"] == 2
    assert t[0]["actions"] == ["go to desk 1", "take cd 1", "use lamp 1"]
    # step1 -> [a1,a2] then terminal empty dropped
    assert t[1]["actions"] == ["take cd 1", "use lamp 1"]
    # step2 -> [a2]; terminal empty turn yields no target (filtered)
    assert t[2]["actions"] == ["use lamp 1"]
    assert len(t) == 3
    print("ok build_plan_targets")


def test_k_window_truncation():
    turns = [("user", "instr"), ("assistant", "OK")]
    for i in range(6):
        turns.append(("user", f"obs{i}"))
        turns.append(("assistant", f"Action: a{i}"))
    c = conv(*turns)
    t = build_plan_targets(c, k=3)
    assert t[0]["actions"] == ["a0", "a1", "a2"]
    assert t[3]["actions"] == ["a3", "a4", "a5"]
    assert t[5]["actions"] == ["a5"]   # last step, only itself
    print("ok k_window")


def test_inline_plan_instruction():
    s = plan_forecast.inline_plan_instruction(3)
    assert "EVERY turn" in s                       # multi-turn compliance emphasis
    assert "Plan:" in s and "Action:" in s         # full-replacement, Plan -> Action
    assert s.index("Plan:") < s.index("Action:")
    assert "1. ...\n2. ...\n3. ..." in s           # K-line skeleton
    assert "1. ...\n2. ...\n3. ...\n4." not in plan_forecast.inline_plan_instruction(3)
    assert plan_forecast.inline_plan_instruction(2).count(". ...") == 2
    # todo style: full replacement with (done) markers
    td = plan_forecast.todo_plan_instruction(3)
    assert "EVERY turn" in td and "(done)" in td and "AVAILABLE ACTIONS" in td
    print("ok inline_plan_instruction")


def test_extract_action_from_block1_turn():
    # block-1 turn: plan listed in the Thought, single real Action: marker.
    turn = ("Thought: My plan for the next 3 actions:\n"
            "1. go to desk 1\n2. take cd 1\n3. use desklamp 1\n\n"
            "Action: go to desk 1")
    assert extract_action(turn) == "go to desk 1"     # picks the real action
    # block-2 target assembly stays correct on a block-1 style trajectory
    c = conv(("user", "instr"), ("assistant", "OK"),
             ("user", "obs0"), ("assistant", turn),
             ("user", "obs1"), ("assistant", "Thought: plan...\nAction: take cd 1"))
    t = build_plan_targets(c, k=3)
    assert t[0]["actions"] == ["go to desk 1", "take cd 1"]
    print("ok extract_action_from_block1_turn")


def test_build_plan_targets_subgoals():
    # TODO-style turns: sub-goals marked (done) hindsight-confirm at first done turn.
    s0 = ("Plan:\n1. find a cd\n2. put cd on desk\nAction:\ngo to shelf 1")
    s1 = ("Plan:\n1. find a cd (done)\n2. put cd on desk\nAction:\ntake cd 1 from shelf 1")
    s2 = ("Plan:\n1. find a cd (done)\n2. put cd on desk (done)\nAction:\nput cd 1 in desk 1")
    c = conv(("user", "instr"), ("assistant", "OK"),
             ("user", "obs0"), ("assistant", s0),
             ("user", "obs1"), ("assistant", s1),
             ("user", "obs2"), ("assistant", s2))
    ach = plan_forecast.achieved_subgoals(c)
    assert [a[0] for a in ach] == ["find a cd", "put cd on desk"]
    assert ach[0][1] == 1 and ach[1][1] == 2   # step indices where first marked done
    t = build_plan_targets(c, k=3)
    # step0: subgoals achieved at step>=0 -> both; step2: only the one done at step2
    assert t[0]["subgoals"] == ["find a cd", "put cd on desk"]
    assert t[2]["subgoals"] == ["put cd on desk"]
    print("ok build_plan_targets_subgoals")


class _FakeTok:
    """Minimal whitespace tokenizer; renders chat with explicit role tags so the
    prefix is a strict text-prefix of the full sequence (clean token boundary)."""
    pad_token_id = 0

    def __init__(self):
        self._v = {"<pad>": 0}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        parts = [f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in msgs]
        if add_generation_prompt:
            parts.append("<assistant>\n")
        return "\n".join(parts) if not add_generation_prompt else \
            "\n".join(parts[:-1]) + ("\n" if parts[:-1] else "") + parts[-1]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        import torch
        ids = []
        for tok in text.split():
            if tok not in self._v:
                self._v[tok] = len(self._v)
            ids.append(self._v[tok])
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids):
        inv = {v: k for k, v in self._v.items()}
        return " ".join(inv[int(i)] for i in ids)


def _traj():
    return conv(("user", "instr"), ("assistant", "OK"),
                ("user", "obs0"), ("assistant", "Thought: t\nAction:\ngo to desk 1"),
                ("user", "obs1"), ("assistant", "Thought: t\nAction:\ntake cd 1"),
                ("user", "obs2"), ("assistant", "Thought: t\nAction:\nuse lamp 1"))


def _masked_text(tok, sample):
    ids = sample["input_ids"][sample["loss_mask"].bool()]
    return tok.decode(ids)


def test_samples_seq_separate():
    tok = _FakeTok()
    samples = plan_forecast.build_plan_forecast_samples(
        _traj(), tok, k=3, target="action", seq="separate")
    assert len(samples) == 3
    tgt = _masked_text(tok, samples[0])
    # bare action list, NO Plan:/Action: scaffolding in the target
    assert "go" in tgt and "desk" in tgt and "Plan:" not in tgt and "Action:" not in tgt
    # synthetic plan prompt appears somewhere in the full sequence (prefix)
    full = tok.decode(samples[0]["input_ids"])
    assert "list" in full and "actions" in full
    print("ok samples_seq_separate")


def test_samples_seq_inline_consistent():
    tok = _FakeTok()
    samples = plan_forecast.build_plan_forecast_samples(
        _traj(), tok, k=3, target="action", seq="inline_consistent")
    assert len(samples) == 3
    tgt = _masked_text(tok, samples[0])
    # target == a real rollout turn: Plan block + Action line
    assert "Plan:" in tgt and "Action:" in tgt and "go" in tgt
    # NO synthetic "list the next K" prompt anywhere (prefix is just the obs)
    full = tok.decode(samples[0]["input_ids"])
    assert "list" not in full
    print("ok samples_seq_inline_consistent")


def test_samples_target_subgoal():
    tok = _FakeTok()
    c = conv(("user", "instr"), ("assistant", "OK"),
             ("user", "obs0"), ("assistant", "Plan:\n1. find a cd\nAction:\ngo to shelf 1"),
             ("user", "obs1"), ("assistant", "Plan:\n1. find a cd (done)\nAction:\ntake cd 1"))
    samples = plan_forecast.build_plan_forecast_samples(
        c, tok, k=3, target="subgoal", seq="separate")
    assert len(samples) >= 1
    tgt = _masked_text(tok, samples[0])
    assert "find" in tgt and "cd" in tgt
    print("ok samples_target_subgoal")


def test_parse_k_schedule():
    ps = plan_forecast.parse_k_schedule
    assert ps("") == [] and ps(None) == []
    assert ps("0:2:3,40:2:4,80:3:5") == [(0, 2, 3), (40, 2, 4), (80, 3, 5)]
    # tolerates spaces + unsorted input (sorted by start)
    assert ps(" 80:3:5 , 0:2:3 , 40:2:4 ") == [(0, 2, 3), (40, 2, 4), (80, 3, 5)]
    for bad in ["40:2:3",            # first stage not at 0
                "0:3:2",             # kMin > kMax
                "0:0:3",             # kMin < 1
                "0:2:3,0:2:4",       # non-increasing starts
                "0:2",               # wrong field count
                "0:a:3"]:            # non-integer
        try:
            ps(bad); assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass
    print("ok parse_k_schedule")


def test_active_k_range():
    st = plan_forecast.parse_k_schedule("0:2:3,40:2:4,80:3:5")
    akr = plan_forecast.active_k_range
    assert akr(st, 0) == (2, 3) and akr(st, 39) == (2, 3)
    assert akr(st, 40) == (2, 4) and akr(st, 79) == (2, 4)
    assert akr(st, 80) == (3, 5) and akr(st, 100000) == (3, 5)
    print("ok active_k_range")


def _traj_n(n):
    turns = [("user", "instr"), ("assistant", "OK")]
    for i in range(n):
        turns.append(("user", f"obs{i}"))
        turns.append(("assistant", f"Thought: t{i}\nAction:\ngo to place {i}"))
    return conv(*turns)


def test_schedule_fixed_range_clamps():
    tok = _FakeTok()
    c = _traj_n(6)   # 6 action turns
    s = plan_forecast.build_plan_forecast_samples(
        c, tok, target="action", seq="separate", k_min=3, k_max=3)
    # k_min==k_max==3 -> realized = min(3, remaining): [3,3,3,3,2,1]
    assert [x['k_realized'] for x in s] == [3, 3, 3, 3, 2, 1]
    print("ok schedule_fixed_range_clamps")


def test_schedule_mixed_draw_bounds_and_alignment():
    import random as _r
    tok = _FakeTok()
    n = 10
    c = _traj_n(n)
    rng = _r.Random(0)
    s = plan_forecast.build_plan_forecast_samples(
        c, tok, target="action", seq="separate", k_min=2, k_max=4, rng=rng)
    assert len(s) == n
    fullhorizon = []
    for i, smp in enumerate(s):
        remaining = n - i
        kr = smp['k_realized']
        assert 1 <= kr <= min(4, remaining), (i, kr, remaining)
        if remaining >= 4:
            assert 2 <= kr <= 4
            fullhorizon.append(kr)
        # prompt aligned: the separate synthetic prompt names the realized count
        full = tok.decode(smp['input_ids'])
        assert f"next {kr} actions" in full
    assert len(set(fullhorizon)) >= 2, "expected a mix of horizons across the batch"
    print("ok schedule_mixed_draw_bounds_and_alignment")


def test_is_invalid_outcome_per_env():
    iio = plan_forecast.is_invalid_outcome
    # common: "Invalid Action." across all envs
    for e in ("alfworld", "sciworld", "webshop", "babyai", "unknown"):
        assert iio("Invalid Action.\n\n...", e)
    # alfworld engine no-op
    assert iio("Nothing happens.", "alfworld")
    # sciworld engine invalid
    assert iio("No known action matches that input.", "sciworld")
    # per-env specificity: 'Nothing happens' is alfworld-only, NOT sciworld
    assert not iio("Nothing happens.", "sciworld")
    assert not iio("No known action matches that input.", "alfworld")
    # normal obs -> valid
    assert not iio("You arrive at desk 1. On it you see a cd 1.", "alfworld")
    print("ok is_invalid_outcome_per_env")


def test_skip_invalid_drops_ineffective_actions():
    c = conv(("user", "instr"), ("assistant", "OK"),
             ("user", "obs0"), ("assistant", "Action:\ngo to desk 1"),
             ("user", "You arrive at desk 1."), ("assistant", "Action:\ntake cd 1"),
             ("user", "Nothing happens."), ("assistant", "Action:\ntake cd 2"),
             ("user", "You pick up the cd 2."), ("assistant", "Action:\nput cd 2 in desk 1"),
             ("user", "You put the cd 2 in the desk 1."))
    # OFF: raw next-3 incl. the ineffective 'take cd 1'
    t_off = build_plan_targets(c, k=3, skip_invalid=False)
    assert t_off[0]["actions"] == ["go to desk 1", "take cd 1", "take cd 2"]
    # ON (alfworld): 'take cd 1' (-> "Nothing happens.") dropped, look past it
    t_on = build_plan_targets(c, k=3, skip_invalid=True, env="alfworld")
    assert t_on[0]["actions"] == ["go to desk 1", "take cd 2", "put cd 2 in desk 1"]
    assert all("take cd 1" not in tgt["actions"] for tgt in t_on)
    # env specificity: for sciworld, "Nothing happens." is NOT an invalid marker,
    # so 'take cd 1' is kept
    t_sci = build_plan_targets(c, k=3, skip_invalid=True, env="sciworld")
    assert t_sci[0]["actions"] == ["go to desk 1", "take cd 1", "take cd 2"]
    print("ok skip_invalid_drops_ineffective_actions")


def test_group_weight_norm():
    tok = _FakeTok()
    traj = _traj_n(2)                 # 2 samples per trajectory
    ml = [traj] * 9
    rewards = [1, 0, 0,  1, 1, 0,  1, 1, 1]   # successes: L x1, M x2, H x3
    gids = ['L', 'L', 'L', 'M', 'M', 'M', 'H', 'H', 'H']
    batch, m = plan_forecast.build_plan_forecast_batch(
        ml, tok, rewards=rewards, target="action", seq="separate",
        group_ids=gids, group_norm=True)
    # all successful trajectories distilled (wins-only), across 3 groups
    assert m["plan_forecast/n_traj_considered"] == 6
    assert m["plan_forecast/group_norm"] == 1.0 and m["plan_forecast/group_n_distilled"] == 3
    lw = batch["loss_weight"]
    assert lw.shape[0] == 12                       # 6 trajs x 2 samples
    # per-traj weight = (1/#success-in-group), normalized to mean 1:
    # raw L=1/1, M=1/2, H=1/3 ; sample order L0(2), M(4), H(6)
    assert abs(lw[:2].mean().item() - 2.0) < 1e-4      # L
    assert abs(lw[2:6].mean().item() - 1.0) < 1e-4     # M
    assert abs(lw[6:].mean().item() - 2.0 / 3.0) < 1e-4  # H
    assert abs(lw.mean().item() - 1.0) < 1e-4          # normalized mean 1
    # KEY: every group contributes the SAME total weight (stability invariant)
    tot_L, tot_M, tot_H = lw[:2].sum().item(), lw[2:6].sum().item(), lw[6:].sum().item()
    assert abs(tot_L - tot_M) < 1e-3 and abs(tot_M - tot_H) < 1e-3
    print("ok group_weight_norm")


def test_group_norm_edge_no_success():
    tok = _FakeTok()
    # (a) whole batch has no success -> None batch, no crash / no div-by-zero
    ml = [_traj_n(2)] * 4
    batch, m = plan_forecast.build_plan_forecast_batch(
        ml, tok, rewards=[0, 0, 0, 0], target="action", seq="separate",
        group_ids=['a', 'a', 'b', 'b'], group_norm=True)
    assert batch is None
    assert m["plan_forecast/n_traj_considered"] == 0.0
    assert m["plan_forecast/group_norm"] == 1.0
    assert "plan_forecast/group_n_distilled" not in m   # skipped, no empty-dict math
    # (b) one group all-fail (A), the other has a success (B): only B distilled
    batch, m = plan_forecast.build_plan_forecast_batch(
        ml, tok, rewards=[0, 0, 1, 0], target="action", seq="separate",
        group_ids=['A', 'A', 'B', 'B'], group_norm=True)
    assert m["plan_forecast/n_traj_considered"] == 1.0        # only B's 1 success
    assert m["plan_forecast/group_n_distilled"] == 1.0        # group A absent (no success)
    assert abs(batch["loss_weight"].mean().item() - 1.0) < 1e-4
    print("ok group_norm_edge_no_success")


def _grp_ml():
    tok = _FakeTok()
    ml = [_traj_n(2)] * 9
    rewards = [1, 0, 0,  1, 1, 0,  1, 1, 1]   # successes: L x1, M x2, H x3
    gids = ['L', 'L', 'L', 'M', 'M', 'M', 'H', 'H', 'H']
    return tok, ml, rewards, gids


def test_group_gating():
    tok, ml, rewards, gids = _grp_ml()

    def considered(**kw):
        _b, m = plan_forecast.build_plan_forecast_batch(
            ml, tok, rewards=rewards, target="action", seq="separate",
            group_ids=gids, **kw)
        return m["plan_forecast/n_traj_considered"], m

    c, _ = considered(gate="wins")                                   # gating off -> 6
    assert c == 6
    c, m = considered(group_gate="low", group_low=0.5, group_high=1.0)   # only L(0.33)
    assert c == 1 and m["plan_forecast/group_n_kept"] == 1 and m["plan_forecast/group_gate"] == 1.0
    c, m = considered(group_gate="low_high", group_low=0.5, group_high=1.0)  # L + H
    assert c == 4 and m["plan_forecast/group_n_kept"] == 2 and m["plan_forecast/group_gate"] == 2.0
    print("ok group_gating")


def test_group_gate_and_norm_compose():
    tok, ml, rewards, gids = _grp_ml()
    batch, m = plan_forecast.build_plan_forecast_batch(
        ml, tok, rewards=rewards, target="action", seq="separate", group_ids=gids,
        group_gate="low_high", group_low=0.5, group_high=1.0, group_norm=True)
    # gating keeps L(1) + H(3); M filtered. norm reweights the kept ones per group.
    assert m["plan_forecast/n_traj_considered"] == 4
    assert m["plan_forecast/group_gate"] == 2.0 and m["plan_forecast/group_norm"] == 1.0
    assert m["plan_forecast/group_n_distilled"] == 2          # L, H only
    lw = batch["loss_weight"]                                  # L 2 samples, H 6 samples
    assert abs(lw[:2].sum().item() - lw[2:].sum().item()) < 1e-3   # equal group totals
    assert abs(lw.mean().item() - 1.0) < 1e-4
    print("ok group_gate_and_norm_compose")


def _traj_acts(acts):
    turns = [("user", "instr"), ("assistant", "OK")]
    for i, a in enumerate(acts):
        turns.append(("user", f"obs{i}"))
        turns.append(("assistant", f"Thought: t{i}\nAction:\n{a}"))
    return conv(*turns)


def test_group_dedup():
    tok = _FakeTok()
    A = _traj_acts(["go north", "go south"])     # seq A
    B = _traj_acts(["open door", "take key"])    # seq B (distinct)
    # one group: 3 copies of A + 1 of B, all successful. k=1 -> 2 samples/traj.
    ml = [A, A, A, B]; rewards = [1, 1, 1, 1]; gids = ["g", "g", "g", "g"]

    def run(dedup):
        b, m = plan_forecast.build_plan_forecast_batch(
            ml, tok, rewards=rewards, k=1, target="action", seq="separate",
            group_ids=gids, group_norm=True, group_dedup=dedup)
        return b["loss_weight"], m

    # order of samples: A1,A2,A3,B -> lw[:6] = A (3 trajs x2), lw[6:] = B (1 traj x2)
    lw, m = run(True)
    assert lw.shape[0] == 8, lw.shape[0]
    # DEDUP: each distinct seq gets equal TOTAL weight -> A-total == B-total
    assert abs(float(lw[:6].sum()) - float(lw[6:].sum())) < 1e-4
    # per-copy: B (1 copy) is 3x an A-copy (3 copies share A's share)
    assert abs(float(lw[6:].mean()) / float(lw[:6].mean()) - 3.0) < 1e-4
    assert abs(float(lw.mean()) - 1.0) < 1e-4
    assert m["plan_forecast/group_dedup"] == 1.0
    assert abs(m["plan_forecast/group_unique_frac"] - 0.5) < 1e-6   # 2 unique / 4 trajs

    # LEGACY (dedup off): per-trajectory -> all equal, A-total : B-total = 3 : 1
    lw2, m2 = run(False)
    assert abs(float(lw2.max()) - float(lw2.min())) < 1e-4          # all weights equal
    assert abs(float(lw2[:6].sum()) / float(lw2[6:].sum()) - 3.0) < 1e-4
    assert m2["plan_forecast/group_dedup"] == 0.0
    print("ok group_dedup")


if __name__ == "__main__":
    test_extract_action()
    test_action_turn_indices_skips_ack()
    test_build_plan_targets_includes_current_and_window()
    test_k_window_truncation()
    test_inline_plan_instruction()
    test_extract_action_from_block1_turn()
    test_build_plan_targets_subgoals()
    test_samples_seq_separate()
    test_samples_seq_inline_consistent()
    test_samples_target_subgoal()
    test_parse_k_schedule()
    test_active_k_range()
    test_schedule_fixed_range_clamps()
    test_schedule_mixed_draw_bounds_and_alignment()
    test_is_invalid_outcome_per_env()
    test_skip_invalid_drops_ineffective_actions()
    test_group_weight_norm()
    test_group_norm_edge_no_success()
    test_group_gating()
    test_group_gate_and_norm_compose()
    test_group_dedup()
    print("ALL PASS")
