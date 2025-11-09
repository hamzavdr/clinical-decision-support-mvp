# scripts/parse_pdf.py
import fitz, re, json
from pathlib import Path

RE_TOP = re.compile(r"^(\d+\.\d+)\s+([A-Z][A-Z\s\(\)\-/]+)$")
RE_SUB = re.compile(r"^(\d+\.\d+\.\d+)\s+(.+)$")
RE_ALLCAPS = re.compile(r"^[A-Z][A-Z\s\-/\(\)\.:0-9]+$")
RE_GRADE = re.compile(r"^Grade\s+([1-4])$", re.IGNORECASE)

def normalize_lines(s: str) -> list[str]:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)           # trim trail spaces
    s = re.sub(r"(\S)-\n(\S)", r"\1\2", s)     # de-hyphenate wraps
    s = re.sub(r"\n{3,}", "\n\n", s)           # collapse blank lines
    return [ln.rstrip() for ln in s.split("\n")]

def parse(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    blocks, ctx = [], dict(section_num=None, section_title=None,
                           subsection_num=None, subsection_title=None,
                           heading=None, grade=None)
    buf, page_start = [], None

    def flush(page_end: int):
        nonlocal buf, page_start
        if not buf: return
        text = "\n".join(buf).strip()
        if text:
            blocks.append({
                "text": text, "page_start": page_start or page_end, "page_end": page_end, **ctx
            })
        buf, page_start = [], None

    for p in range(len(doc)):
        lines = normalize_lines(doc.load_page(p).get_text("text"))
        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            m_top, m_sub, m_grade = RE_TOP.match(ln), RE_SUB.match(ln), RE_GRADE.match(ln)

            if m_top:
                flush(p+1)
                ctx.update(section_num=m_top.group(1), section_title=m_top.group(2).strip(),
                           subsection_num=None, subsection_title=None, heading=None, grade=None)
                i += 1; continue
            if m_sub:
                flush(p+1)
                ctx.update(subsection_num=m_sub.group(1), subsection_title=m_sub.group(2).strip(),
                           heading=None, grade=None)
                i += 1; continue
            if RE_ALLCAPS.match(ln) and len(ln.split()) <= 5:
                flush(p+1)
                ctx.update(heading=ln.strip(), grade=None)
                i += 1; continue
            if m_grade:
                flush(p+1)
                ctx.update(heading="GRADE", grade=m_grade.group(1))
                i += 1; continue

            if page_start is None: page_start = p+1
            buf.append(ln); i += 1

    flush(len(doc)); doc.close(); return blocks

if __name__ == "__main__":
    out = parse(Path("data/guidelines.pdf"))
    Path("data/blocks.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {len(out)} blocks → data/blocks.json")
