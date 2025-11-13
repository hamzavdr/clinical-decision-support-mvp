#!/usr/bin/env python3
"""
assign_page_metadata_to_json.py

Take a JSON file of pages like:

[
  {
    "page_number": 1,
    "orientation": "portrait",
    "page_header": "...",
    "page_body": "...",
    "page_footer": "1.1"
  },
  ...
]

and add:
  - chapter_number      (int or null)
  - chapter_page_index  (int or null)
  - page_type           ("chapter_contents" | "chapter_body" | "chapter_references")

Just edit INPUT_JSON and OUTPUT_JSON below.
"""

import json
import re
from collections import defaultdict

# ----------------------------------------------------------------------
# CONFIG: set these two paths for your files
# ----------------------------------------------------------------------
INPUT_JSON  = "guideline_documents/sa-phc-stg-2024_pages.json"
OUTPUT_JSON = "guideline_documents/sa-phc-stg-2024_pages_with_metadata.json"
# ----------------------------------------------------------------------

# Detect chapter-local page numbers in the footer like "1.1", "10.2", etc.
FOOTER_CHAPTER_PATTERN = re.compile(r"\b(\d+)\.(\d+)\b")


def add_chapter_indices(pages):
    """
    For each page, look for 'X.Y' in the footer and store:
      - chapter_number = X
      - chapter_page_index = Y
    If no match, both are None.
    """
    for page in pages:
        footer = page.get("page_footer", "") or ""
        m = FOOTER_CHAPTER_PATTERN.search(footer)
        if m:
            page["chapter_number"] = int(m.group(1))
            page["chapter_page_index"] = int(m.group(2))
        else:
            page["chapter_number"] = None
            page["chapter_page_index"] = None


def assign_page_types(pages):
    """
    Adds page_type to each page:
      - 'chapter_contents'
      - 'chapter_body'
      - 'chapter_references'

    Logic (per chapter_number):
      - Group pages by chapter_number.
      - Sort by chapter_page_index (fallback to global page_number).
      - If any page in the chapter contains "references"/"bibliography":
          * Let first_ref_index = smallest chapter_page_index with that keyword.
          * All pages with chapter_page_index >= first_ref_index => chapter_references.
        Else:
          * Last 1–2 pages in the chapter => chapter_references.
      - First 1–2 chapter-local pages (1 or 2), excluding references => chapter_contents.
      - Everything else => chapter_body.

    Pages with no chapter_number are defaulted to 'chapter_body'.
    """

    # Group by chapter_number (ignore pages with no chapter_number)
    chapters = defaultdict(list)
    for p in pages:
        ch = p.get("chapter_number")
        if ch is not None:
            chapters[ch].append(p)

    for ch_num, ch_pages in chapters.items():
        # Sort pages within chapter by chapter_page_index, then by global page_number
        ch_pages.sort(
            key=lambda p: (
                p.get("chapter_page_index")
                if p.get("chapter_page_index") is not None
                else 10**9,
                p.get("page_number", 10**9),
            )
        )
        n = len(ch_pages)

        # 1) Detect initial refs by keyword
        for p in ch_pages:
            text = (p.get("page_header", "") + "\n" + p.get("page_body", "")).lower()
            if "references" in text or "bibliography" in text:
                p["page_type"] = "chapter_references"
            else:
                p["page_type"] = None  # unset for now

        ref_pages = [p for p in ch_pages if p["page_type"] == "chapter_references"]

        # 2) If we found any reference page, treat everything from the first such page
        #    to the end of the chapter as references.
        if ref_pages:
            # smallest chapter_page_index among reference pages
            first_ref_index = min(
                rp.get("chapter_page_index", 10**9) for rp in ref_pages
            )

            for p in ch_pages:
                idx = p.get("chapter_page_index")
                if idx is not None and idx >= first_ref_index:
                    p["page_type"] = "chapter_references"

        # 3) If we didn't find any refs at all, fallback: make last 1–2 pages references
        if not ref_pages:
            if n >= 2:
                for p in ch_pages[-2:]:
                    p["page_type"] = "chapter_references"
            elif n == 1:
                # single-page chapter: don't force references
                pass

        # 4) Mark chapter_contents: first 1–2 chapter-local pages (indices 1 or 2)
        #    that are not already references
        for p in ch_pages:
            if p["page_type"] == "chapter_references":
                continue
            idx = p.get("chapter_page_index")
            if idx in (1, 2):
                p["page_type"] = "chapter_contents"

        # 5) Everything else in the chapter is chapter_body
        for p in ch_pages:
            if p["page_type"] is None:
                p["page_type"] = "chapter_body"

    # Pages with no chapter_number: treat as 'chapter_body'
    for p in pages:
        if "page_type" not in p or p["page_type"] is None:
            p["page_type"] = "chapter_body"


def main():
    # Load pages JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Step 1: derive chapter_number and chapter_page_index from footer
    add_chapter_indices(pages)

    # Step 2: assign page_type based on chapter structure
    assign_page_types(pages)

    # Save enhanced JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"Wrote enhanced JSON with metadata to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
