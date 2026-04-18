#!/usr/bin/env python3

import json
from pathlib import Path


def build_rows(start: int, stop: int) -> list[dict]:
    rows = []
    for local_idx, official_idx in enumerate(range(start, stop)):
        rows.append(
            {
                "item_id": f"alfworld_{local_idx}",
                "conversations": [],
                "official_item_id": f"alfworld_{official_idx}",
                "official_goal_idx": official_idx,
            }
        )
    return rows


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    
    # The project's AlfWorld environment contains 2620 games in mappings_train.json and mappings_test.json.
    total = 2620
    test_end = 200
    eval_end = 500

    split_to_rows = {
        "train": build_rows(eval_end, total),
        "eval": build_rows(test_end, eval_end),
        "test": build_rows(0, test_end),
        "all": build_rows(0, total),
    }

    output_root = root / "AgentItemId"
    write_json(output_root / "train" / "alfworld_train.json", split_to_rows["train"])
    write_json(output_root / "eval" / "alfworld_eval.json", split_to_rows["eval"])
    write_json(output_root / "test" / "alfworld_test.json", split_to_rows["test"])
    write_json(output_root / "all" / "alfworld_all.json", split_to_rows["all"])

    print(f"Prepared AlfWorld splits in {output_root}")


if __name__ == "__main__":
    main()
