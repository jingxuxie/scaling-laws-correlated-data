#!/usr/bin/env python3
"""Fill the official AAAI reproducibility checklist without changing questions."""

from __future__ import annotations

import argparse
from pathlib import Path


# Responses follow the order of the question placeholders after the official
# ``% The questions start here`` sentinel.  The template's instructions also
# mention the placeholder text, so counting/replacing over the whole document
# would corrupt the unmodified explanatory preamble.
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
    sentinel = "% The questions start here"

    if sentinel in template:
        preamble, questions = template.split(sentinel, 1)
        prefix = preamble + sentinel
    else:
        # Keep the helper easy to unit-test on a synthetic question-only input.
        prefix = ""
        questions = template

    count = questions.count(marker)
    if count != len(ANSWERS):
        raise ValueError(
            f"expected {len(ANSWERS)} checklist response slots after the "
            f"question sentinel, found {count}; the official template may "
            "have changed"
        )

    completed_questions = questions
    for answer in ANSWERS:
        completed_questions = completed_questions.replace(marker, answer, 1)
    if marker in completed_questions:
        raise AssertionError("unfilled checklist response remains in question block")
    return prefix + completed_questions


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
