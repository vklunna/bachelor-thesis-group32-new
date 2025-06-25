import pandas as pd
import fitz
import re
from pathlib import Path
import unicodedata
import random


def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)

def parse_page_range(text):
    text = str(text)
    numbers = set()
    for part in re.split(r"[;,]", text):
        part = part.strip()
        if re.match(r"^\d+\s*[-–]\s*\d+$", part):
            start, end = map(int, re.split(r"[-–]", part))
            numbers.update(range(start, end + 1))
        elif part.isdigit():
            numbers.add(int(part))
    return numbers

def extract_pdf_text_by_code_and_section(pdf_path):
    pdf_path = Path(pdf_path)
    pdf_stem = pdf_path.stem

    table_path = Path("../2_output/standardized_merged_by_company") / f"{pdf_stem}_standardized_full.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Cannot find the standardized table at: {table_path}")

    df = pd.read_csv(table_path)
    df.columns = df.columns.str.strip()
    df["Code"] = df["Code"].astype(str).str.strip()
    df["Relevant Pages"] = df["Relevant Pages"].astype(str).str.strip()
    df["Section Reference"] = df["Section Reference"].astype(str).str.strip()
    df["Page Range"] = df["Page Range"].astype(str).str.strip()

    doc = fitz.open(pdf_path)
    output = []
    total_pages = len(doc)
    cutoff_page = int(total_pages * 0.8)

    for i, row in df.iterrows():
        code = row["Code"]
        valid_code = bool(re.search(r"\b(ESRS\s*)?E[-\s]?\d+\b", code, re.IGNORECASE))
        if not valid_code:
            continue

        rel_pages_raw = row["Relevant Pages"]
        section_raw = row["Section Reference"]
        page_range_set = parse_page_range(row.get("Page Range", ""))

        # ✅ "Relevant Pages" handling
        used_pages = set()
        if rel_pages_raw and pd.notna(rel_pages_raw):
            page_nums = [int(p) for p in re.findall(r"\d+", rel_pages_raw)]
            for p in page_nums:
                if 21 <= p <= cutoff_page:
                    used_pages.add(p)

        for page_num in sorted(used_pages):
            page_index = page_num - 1  # PDF number → Python index
            raw_text = doc[page_index].get_text()
            page_text = normalize_whitespace(normalize_unicode(raw_text))
            output.append({
                "Matched Code": code if valid_code else None,
                "Relevant Pages (Raw)": rel_pages_raw,
                "Used Pages": page_num,
                "Section Reference": None,
                "Extracted Text": page_text.strip()
            })

        # ✅ "Section Reference" match within pages 21 to cutoff_page
        if section_raw and pd.notna(section_raw):
            section_list = [s.strip() for s in section_raw.split(",") if s.strip()]
            for sec in section_list:
                matched_pages = set()
                for page_index in range(20, cutoff_page):  # Python index for pages 21 to cutoff
                    pdf_page_num = page_index + 1
                    if pdf_page_num in page_range_set:
                        continue
                    page_text = doc[page_index].get_text()
                    match = re.search(rf"\b{re.escape(sec)}\b", page_text)
                    if match:
                        matched_pages.add(page_index)
                        print(f"[Section Match] Row {i}, Code='{code}', Section='{sec}', Triggered Page={pdf_page_num}")

                        # ✅ Extract text starting from match
                        start_index = match.start()
                        trimmed_text = page_text[start_index:]
                        trimmed_text = normalize_whitespace(normalize_unicode(trimmed_text))

                        output.append({
                            "Matched Code": code if valid_code else None,
                            "Relevant Pages (Raw)": rel_pages_raw,
                            "Used Pages": pdf_page_num,
                            "Section Reference": sec,
                            "Extracted Text": trimmed_text.strip()
                        })

                if not matched_pages:
                    print(f"[Section Miss] Row {i}, Code='{code}', Section='{sec}' — no matches found.")

    return pd.DataFrame(output)