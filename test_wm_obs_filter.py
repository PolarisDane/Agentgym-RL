"""Unit tests for the webshop WM-SFT observation-target filter (schemas.py)."""
import importlib.util
import json
import glob
import os
import sys

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "AgentGym-RL/verl/workers/rollout/schemas.py")
_spec = importlib.util.spec_from_file_location("schemas", _p)
schemas = importlib.util.module_from_spec(_spec)
sys.modules["schemas"] = schemas
_spec.loader.exec_module(schemas)
_filter = schemas._webshop_obs_target_mask

from transformers import AutoTokenizer

TOK = ("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/"
       "models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
tok = AutoTokenizer.from_pretrained(TOK)

SEARCH = ("Instruction: [SEP] Find me women's sandals with arch support [SEP] Back to Search "
          "[SEP] Page 1 (Total results: 50) [SEP] Next > "
          "[SEP] B07XDRVVYM [SEP] Clarks Women's Un Adorn Sling Sandal [SEP] $34.99 to $99.95 "
          "[SEP] B09S6VN97V [SEP] Skechers Women's Arch Fit Commute Clog Taupe 11 M [SEP] $74.95")
DETAIL = ("Instruction: [SEP] Find me women's sandals [SEP] Back to Search [SEP] < Prev "
          "[SEP] size [SEP] 8 [SEP] 8 wide [SEP] color [SEP] black combi "
          "[SEP] Clarks Women's Un Adorn Sling Sandal [SEP] Price: $34.99 to $99.95 "
          "[SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Buy Now")


def decode_kept_dropped(content):
    mask, drop, tot, fs = _filter(content, tok)
    ids = tok(content, add_special_tokens=False)["input_ids"]
    kept = tok.decode([i for i, m in zip(ids, mask) if m])
    gone = tok.decode([i for i, m in zip(ids, mask) if not m])
    return mask, drop, tot, fs, kept, gone


def test_search_page_drops_product_fields():
    mask, drop, tot, fs, kept, gone = decode_kept_dropped(SEARCH)
    assert not fs, "should not hit fail-safe on a well-formed search page"
    # product content must be gone
    for s in ["B07XDRVVYM", "B09S6VN97V", "Clarks", "Skechers", "34.99", "74.95"]:
        assert s not in kept, f"{s!r} should have been dropped, kept={kept!r}"
    # skeleton + instruction echo must survive
    for s in ["Instruction", "arch support", "Back to Search", "Total results", "Next"]:
        assert s in kept, f"{s!r} must be kept, kept={kept!r}"
    assert drop / tot > 0.4, f"expected to drop a large share, got {drop}/{tot}"
    print(f"ok search page: dropped {drop}/{tot} = {drop/tot:.0%}")


def test_detail_page_untouched():
    mask, drop, tot, fs, kept, gone = decode_kept_dropped(DETAIL)
    assert drop == 0 and not fs, "detail pages must not be filtered"
    assert all(m == 1 for m in mask)
    print("ok detail page untouched")


def test_malformed_page_failsafe():
    bad = "Instruction: [SEP] x [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] NOT_AN_ASIN [SEP] title"
    mask, drop, tot, fs, kept, gone = decode_kept_dropped(bad)
    assert fs and drop == 0 and all(m == 1 for m in mask), "fail-safe must keep everything"
    print("ok fail-safe keeps everything on malformed page")


def test_ids_unchanged_and_alignment():
    for c in (SEARCH, DETAIL):
        mask, *_ = _filter(c, tok)
        ids = tok(c, add_special_tokens=False)["input_ids"]
        assert len(mask) == len(ids), "mask must align 1:1 with content tokens"
    print("ok mask aligns with input_ids (ids themselves never modified)")


def test_on_real_rollouts():
    files = sorted(glob.glob("runlogs/webshop_grpo_reb_r0_new_20260728_144251/"
                             "rollout_logs/step*/*.json"))[-2:]
    if not files:
        print("skip real-rollout test (no logs)"); return
    recs = []
    for f in files:
        recs.extend(json.load(open(f)))
    tot = drop = fs = 0
    s_tot = s_drop = 0
    for r in recs[:40]:
        for m in r["conversations"][2:]:
            if m["role"] != "user":
                continue
            c = m["content"]
            _, d, t, f_ = _filter(c, tok)
            tot += t; drop += d; fs += int(f_)
            if "Total results:" in c:
                s_tot += t; s_drop += d
    print(f"ok real rollouts: fail-safe hits = {fs}; "
          f"search-page tokens dropped {s_drop}/{s_tot} = {s_drop/s_tot:.1%}; "
          f"global {drop}/{tot} = {drop/tot:.1%} "
          f"(global share varies with how often the agent searches vs clicks)")
    assert fs == 0, "no fail-safe expected on real data"
    # The stable invariant is WITHIN search pages: product content is the bulk of them.
    # The global share swings with the search/click mix, so do not assert on it.
    assert 0.60 < s_drop / s_tot < 0.90, f"expected ~73% of search-page tokens dropped, got {s_drop/s_tot:.1%}"
    assert drop == s_drop, "only search pages may be filtered"


if __name__ == "__main__":
    test_search_page_drops_product_fields()
    test_detail_page_untouched()
    test_malformed_page_failsafe()
    test_ids_unchanged_and_alignment()
    test_on_real_rollouts()
    print("ALL PASS")
