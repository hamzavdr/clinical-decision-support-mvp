# Converts PDF document to JSON; each page is one json object

import fitz  # PyMuPDF
import json
import re

PDF_PATH = "guideline_documents/sa-phc-stg-2024.pdf"
OUTPUT_JSON = "guideline_documents/sa-phc-stg-2024_pages.json"

# --- Fractions for portrait vs landscape ---
HEADER_FRACTION_PORTRAIT = 0.125
FOOTER_FRACTION_PORTRAIT = 0.10

HEADER_FRACTION_LANDSCAPE = 0.11
FOOTER_FRACTION_LANDSCAPE = 0.11

# --- page number detection patterns for footer cleaning ---
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.IGNORECASE),  # "Page 12", "Page 12 of 700"
    re.compile(r"^\d+\s*/\s*\d+$"),                            # "12/700" or "12 / 700"
    re.compile(r"^page\s*\d+$", re.IGNORECASE),                # "Page12"
    re.compile(r"^\d+$"),                                      # bare number "12"
]


def looks_like_page_number(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return any(pat.match(line) for pat in PAGE_NUMBER_PATTERNS)


def clean_footer_lines(lines):
    return [line for line in lines if not looks_like_page_number(line)]


def extract_page_regions(page):
    """
    Extract header, body, and footer text from a single page.

    - Uses page.rotation_matrix so coordinates match visual orientation.
    - Adjusts header/footer fractions depending on whether the page is
      portrait or landscape (horizontal).
    """
    rect = page.rect
    is_landscape = rect.width > rect.height

    if is_landscape:
        header_fraction = HEADER_FRACTION_LANDSCAPE
        footer_fraction = FOOTER_FRACTION_LANDSCAPE
    else:
        header_fraction = HEADER_FRACTION_PORTRAIT
        footer_fraction = FOOTER_FRACTION_PORTRAIT

    page_height = rect.height
    header_limit = page_height * header_fraction
    footer_limit = page_height * (1 - footer_fraction)

    header_lines = []
    body_lines = []
    footer_lines = []

    blocks = page.get_text("blocks")
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

    rot_matrix = page.rotation_matrix

    for block in blocks:
        x0, y0, x1, y1, text, *_ = block
        text = text.strip()
        if not text:
            continue

        # Transform to visually oriented coordinates
        vrect = fitz.Rect(x0, y0, x1, y1) * rot_matrix
        vy0, vy1 = vrect.y0, vrect.y1

        if vy1 <= header_limit:
            header_lines.append(text)
        elif vy0 >= footer_limit:
            footer_lines.append(text)
        else:
            body_lines.append(text)

    footer_lines = clean_footer_lines(footer_lines)

    header_text = "\n".join(header_lines).strip()
    body_text = "\n".join(body_lines).strip()
    footer_text = "\n".join(footer_lines).strip()

    return header_text, body_text, footer_text, is_landscape


def pdf_to_json(pdf_path, output_json_path):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, start=1):
        header, body, footer, is_landscape = extract_page_regions(page)

        page_obj = {
            "page_number": i,
            "orientation": "landscape" if is_landscape else "portrait",
            "page_header": header or "",
            "page_body": body or "",
            "page_footer": footer or "",
        }
        pages.append(page_obj)

    doc.close()

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    pdf_to_json(PDF_PATH, OUTPUT_JSON)
    print("Done!")