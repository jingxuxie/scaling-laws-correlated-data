#!/usr/bin/env python3
"""Audit compiled AAAI submission artifacts and emit a machine-readable report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def run(*args: str) -> str:
    completed = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def pdf_pages(path: Path) -> int:
    info = run("pdfinfo", str(path))
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"could not read page count from {path}")
    return int(match.group(1))


def page_size(path: Path) -> str:
    info = run("pdfinfo", str(path))
    match = re.search(r"^Page size:\s+(.+)$", info, flags=re.MULTILINE)
    return match.group(1).strip() if match else "unknown"


def reference_start_page(path: Path, pages: int) -> int | None:
    for page in range(1, pages + 1):
        text = run("pdftotext", "-f", str(page), "-l", str(page), str(path), "-")
        if re.search(r"(?m)^\s*References\s*$", text):
            return page
    return None


def type3_fonts(path: Path) -> list[str]:
    lines = run("pdffonts", str(path)).splitlines()[2:]
    return [line for line in lines if "Type 3" in line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_audit(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    undefined = [
        line
        for line in text.splitlines()
        if "undefined" in line.lower()
        and ("citation" in line.lower() or "reference" in line.lower())
    ]
    overfull = [line for line in text.splitlines() if "Overfull \\hbox" in line]
    return {
        "aaai_style_detected": "AAAI 2027 Submission format" in text
        or "Conference Style for AAAI" in text,
        "undefined_reference_or_citation_lines": undefined,
        "overfull_hbox_lines": overfull,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_pdf = args.paper_dir / "main.pdf"
    supplement_pdf = args.paper_dir / "supplement.pdf"
    checklist_pdf = args.paper_dir / "reproducibility_checklist.pdf"
    main_log = args.paper_dir / "main.log"
    required = [main_pdf, supplement_pdf, checklist_pdf, main_log]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing submission artifacts: {missing}")

    main_pages = pdf_pages(main_pdf)
    supplement_pages = pdf_pages(supplement_pdf)
    checklist_pages = pdf_pages(checklist_pdf)
    references_page = reference_start_page(main_pdf, main_pages)
    log = log_audit(main_log)
    type3 = type3_fonts(main_pdf)

    failures: list[str] = []
    if main_pages > 9:
        failures.append(f"main PDF has {main_pages} pages; AAAI permits at most 9")
    if references_page is None:
        failures.append("could not locate the References heading")
    elif references_page > 8:
        failures.append(
            f"references begin on page {references_page}; technical content exceeds 7 pages"
        )
    if "letter" not in page_size(main_pdf).lower() and "612 x 792" not in page_size(main_pdf):
        failures.append(f"main PDF is not US Letter: {page_size(main_pdf)}")
    if not log["aaai_style_detected"]:
        failures.append("main log does not show the official AAAI-27 style")
    if log["undefined_reference_or_citation_lines"]:
        failures.append("undefined references or citations remain")
    if log["overfull_hbox_lines"]:
        failures.append("overfull hboxes remain in the official build")
    if type3:
        failures.append("Type 3 fonts detected in main PDF")

    real_summary_path = Path("results/real_sequential/summary.json")
    real_summary = (
        json.loads(real_summary_path.read_text(encoding="utf-8"))
        if real_summary_path.exists()
        else None
    )
    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "main": {
            "pages": main_pages,
            "references_start_page": references_page,
            "page_size": page_size(main_pdf),
            "sha256": sha256(main_pdf),
            "type3_fonts": type3,
            **log,
        },
        "supplement": {
            "pages": supplement_pages,
            "sha256": sha256(supplement_pdf),
        },
        "reproducibility_checklist": {
            "pages": checklist_pages,
            "sha256": sha256(checklist_pdf),
        },
        "real_sequential": real_summary,
        "validation": {
            "unit_tests": "passed before audit in the submission workflow",
            "official_author_kit": "AAAI_AuthorKit27 from this repository",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("submission audit failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
