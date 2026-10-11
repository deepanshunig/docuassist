"""Validate evaluation IDs, split isolation and source page references."""
import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    seen_ids = set()
    group_splits = {}
    for split, expected_count in (("dev", 20), ("final", 10)):
        records = [json.loads(line) for line in
                   (ROOT / "evaluation" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(records) == expected_count
        for record in records:
            assert record["id"] not in seen_ids, record["id"]
            seen_ids.add(record["id"])
            group = record["group_id"]
            assert group_splits.setdefault(group, split) == split, f"Group crosses splits: {group}"
            assert bool(record["support"]) == record["answerable"]
            for source in record["support"]:
                pages = PdfReader(ROOT / "sample_docs" / source["document"]).pages
                assert 1 <= source["pdf_page"] <= len(pages)
                assert pages[source["pdf_page"] - 1].extract_text().strip()
    print(f"Validated {len(seen_ids)} questions, split isolation and PDF source references")


if __name__ == "__main__":
    main()
