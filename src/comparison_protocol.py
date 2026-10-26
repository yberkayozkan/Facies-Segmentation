"""Shared evaluation contract for the three Parihaka comparison notebooks."""

import json
from pathlib import Path


EXPECTED_SPLIT = {
    "axis": "inline",
    "shape": [590, 782, 1006],
    "train_ranges": [[39, 99], [146, 370], [480, 550]],
    "buffer_ranges": [[31, 39], [99, 107], [138, 146], [370, 378], [472, 480], [550, 558]],
    "val": [378, 472],
    "test_ranges": [[0, 31], [107, 138], [558, 590]],
}


def verify_comparison_split(split: dict) -> None:
    """Fail early if a comparison run uses a different physical data split."""
    differences = {
        key: (split.get(key), expected)
        for key, expected in EXPECTED_SPLIT.items()
        if split.get(key) != expected
    }
    if differences:
        raise ValueError(f"Comparison split differs from the shared protocol: {differences}")


def verify_evaluation_indices(split: dict, validation_indices: list[int], path: str | Path) -> None:
    """Persist the first run's validation selection and check every later run."""
    verify_comparison_split(split)
    expected_test = [idx for start, end in split["test_ranges"] for idx in range(start, end)]
    selection = {
        "shape": split["shape"],
        "validation_inlines": [int(index) for index in validation_indices],
        "test_inlines": expected_test,
    }
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as file:
            saved = json.load(file)
        if saved != selection:
            raise ValueError(f"Evaluation inlines differ from the shared selection in {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(selection, file, indent=2)
