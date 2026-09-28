#!/usr/bin/env python3
# mypy: disable-error-code="no-untyped-call"
"""把候选论文的代表页面渲染为人工版面复核联系表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arxiv_ids", nargs="+", help="需要复核的无版本 arXiv ID")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp/pdfs/pilot_candidates",
    )
    return parser.parse_args()


def _index_pdfs() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for metadata_path in sorted((PROJECT_ROOT / "data/raw/metadata").glob("*.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        arxiv_id = str(payload.get("arxiv_id", ""))
        pdf_name = Path(str(payload.get("pdf_path", ""))).name
        if arxiv_id and pdf_name:
            index[arxiv_id] = PROJECT_ROOT / "data/raw/papers" / pdf_name
    return index


def render_candidate(pdf_path: Path, arxiv_id: str, output_dir: Path) -> Path:
    """在一张高清 PNG 上并排显示最多四个代表页。"""

    source = pymupdf.open(pdf_path)
    try:
        page_numbers = sorted(
            {
                0,
                min(2, source.page_count - 1),
                min(4, source.page_count - 1),
                min(6, source.page_count - 1),
            }
        )
        canvas = pymupdf.open()
        sheet = canvas.new_page(width=1684, height=520)
        sheet.insert_text(
            (24, 28), f"{arxiv_id} | {pdf_path.name} | pages={source.page_count}", fontsize=14
        )
        width = (1640 - 18 * (len(page_numbers) - 1)) / len(page_numbers)
        for column, page_number in enumerate(page_numbers):
            x0 = 22 + column * (width + 18)
            rect = pymupdf.Rect(x0, 48, x0 + width, 500)
            sheet.show_pdf_page(rect, source, page_number, keep_proportion=True)
            sheet.insert_text((x0, 516), f"page {page_number + 1}", fontsize=10)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{arxiv_id.replace('.', '_')}.png"
        sheet.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(output_path)
        canvas.close()
        return output_path
    finally:
        source.close()


def main() -> int:
    args = parse_args()
    index = _index_pdfs()
    missing = [item for item in args.arxiv_ids if item not in index]
    if missing:
        raise SystemExit(f"unknown arXiv IDs: {', '.join(missing)}")
    for arxiv_id in args.arxiv_ids:
        print(render_candidate(index[arxiv_id], arxiv_id, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
