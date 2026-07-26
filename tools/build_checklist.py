#!/usr/bin/env python3
"""Fill the official AAAI reproducibility checklist without modifying its questions."""

from __future__ import annotations

import argparse
from pathlib import Path


# Responses follow the order of ``Type your response here`` occurrences in the
# unmodified AAAI-27 ReproducibilityChecklist.tex file.
ANSWERS = [
    # General paper structure.
    "yes",
    "yes",
    "yes",
    # Theoretical contributions.
    "yes",
    "yes",
    "yes",
    "yes",
    "yes",
    "yes",
    "partial",
    "yes",
    # Dataset usage.
    "yes",
    "yes",
    "NA",
    "NA",
    "yes",
    "yes",
    "NA",
    # Computational experiments.
    "yes",
    "partial",
    "partial",
    "yes",
    "partial",
    "yes",
    "yes",
    "partial",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
]


def fill_checklist(template: str) -> str:
    marker = "Type your response here"
    count = template.count(marker)
    if count != len(ANSWERS):
        raise ValueError(
            f"expected {len(ANSWERS)} checklist response slots, found {count}; "
            "the official template may have changed"
        )
    output = template
    for answer in ANSWERS:
        output = output.replace(marker, answer, 1)
    if marker in output:
        raise AssertionError("unfilled checklist response remains")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.template.read_text(encoding="utf-8")
    completed = fill_checklist(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(completed, encoding="utf-8")
    print(f"wrote {args.output} with {len(ANSWERS)} completed responses")


if __name__ == "__main__":
    main()
