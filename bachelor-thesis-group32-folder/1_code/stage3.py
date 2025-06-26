import os
import pandas as pd
import re
from collections import defaultdict
import re
from pathlib import Path
import unicodedata
from unidecode import unidecode  # for transliteration like é → e

# Regex patterns
combined_code_pattern = re.compile(
    r'^(ESRS\s*\d?\s*[A-Z]{2,4}\s*[-.]\d+(?:\.\d+)?[a-zA-Z]?)\s*[-–—]\s*(.*)',
    re.IGNORECASE
)

esrs_code_pattern = re.compile(
    r'^(ESRS\s*\d?\s*)?[A-Z]{2,4}\s*[-.]\d+(?:\.\d+)?[a-zA-Z]?$',
    re.IGNORECASE
)

extended_code_pattern = re.compile(
    r'^(ESRS\s*\d?\s*)?[A-Z]{2,4}\s*[-.]\d+(?:\.\d+)?[a-zA-Z]?$',
    re.IGNORECASE
)


import re
from unidecode import unidecode


def extract_code_chunks(text):
    """
    Extract ESRS-like codes from text.
    Handles cases like:
    - E1-1, E1 -1
    - GOV-2, SBM -1
    - ESRS 2 E1-3, ESRS2 G1 -1
    - E2-SBM-1
    """
    if not isinstance(text, str) or not text.strip():
        return []

    text = re.sub(r'[‐‒–—―−－⁻]', '-', text)
    text = text.strip()

    # ✅ Limit to allowed ESRS prefixes
    allowed_prefixes = r'(?:E|S|G|GOV|SBM|IRO|BP)'
    code_pattern = rf'((?:ESRS\s*\d?\s*)?(?:{allowed_prefixes})(?:[-.]?\d+)*(?:[-.]\d+(?:\.\d+)?[a-zA-Z]?)*)'

    pattern = re.compile(
        rf'(?:^|\|\s*|\s+)'   # Start of line or separator
        rf'{code_pattern}'    # The actual code
        rf'(?=\s|\||$)',      # End of code
        re.IGNORECASE
    )

    matches = []
    last_end = 0
    for match in pattern.finditer(text):
        full_code = match.group(1).strip()
        full_code = re.sub(r'\s*-\s*', '-', full_code)  # Normalize spacing around dashes
        last_end = match.end()
        matches.append((full_code, ''))

    if matches:
        trailing = text[last_end:].strip().lstrip('|').strip()
        matches[-1] = (matches[-1][0], trailing)

    return matches

def clean_extracted_content(df):
    """Clean DataFrame after code extraction"""
    if not isinstance(df, pd.DataFrame):
        return df
        
    df = df.copy()
    for idx, row in df.iterrows():
        for col in ['Name', 'Description']:
            text = str(row[col])
            for code, _ in extract_code_chunks(text):
                text = re.sub(rf'\s*{re.escape(code)}\s*[\|]*\s*', ' ', text)
            df.at[idx, col] = text.strip()
    return df

def split_rows_on_pipe_with_code(df):
    """Split rows with pipe-separated codes"""
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
        
    expanded_rows = []
    for _, row in df.iterrows():
        parts = [p.strip() for p in str(row['Name']).split('|') if p.strip()]
        for part in parts:
            new_row = row.copy()
            codes = extract_code_chunks(part)
            if codes:
                for code, remaining in codes:
                    new_row['Code'] = code
                    new_row['Name'] = remaining
                    expanded_rows.append(new_row.copy())
            else:
                new_row['Name'] = part
                expanded_rows.append(new_row.copy())
    return pd.DataFrame(expanded_rows)


def process_dataframe(input_df):
    """Main processing pipeline"""
    if not isinstance(input_df, pd.DataFrame):
        return pd.DataFrame()
        
    # Clean content first
    df_clean = clean_extracted_content(input_df)
    
    # Process Description column
    desc_expanded = []
    for _, row in df_clean.iterrows():
        desc_chunks = extract_code_chunks(row['Description'])
        if not desc_chunks:
            desc_expanded.append(row.to_dict())
        else:
            for code, context in desc_chunks:
                new_row = row.copy()
                new_row['Code'] = code
                new_row['Description'] = context
                desc_expanded.append(new_row)
    
    # Process Name column
    name_df = split_rows_on_pipe_with_code(pd.DataFrame(desc_expanded))
    return name_df
    
# Helper: split "code name" if fused (fallback)
def split_code_and_name(text):
    codes = extract_code_chunks(text)
    if codes:
        code = codes[0][0]
        remaining = text.replace(code, '').strip()
        return code, remaining
    return "", text.strip()



def split_rows_on_any_embedded_esrs_code(df):
    """Split rows where Name or Description contain multiple ESRS codes (with or without pipes)."""
    expanded_rows = []

    for _, row in df.iterrows():
        original_code = row.get("Code", "")
        page_range = row.get("Page Range", "")
        relevant_pages = row.get("Relevant Pages", "")
        name_text = str(row.get("Name", "") or "")
        description_text = str(row.get("Description", "") or "")

        # Extract chunks with codes from both fields
        name_chunks = extract_code_chunks(name_text)
        desc_chunks = extract_code_chunks(description_text)

        # If no multiple codes → keep row as is
        if len(name_chunks) + len(desc_chunks) <= 1:
            expanded_rows.append({
                "Code": original_code,
                "Name": name_text,
                "Page Range": page_range,
                "Description": description_text,
                "Relevant Pages": relevant_pages
            })
            continue

        # Expand Name-based codes
        for code, _ in name_chunks:
            cleaned_name = re.sub(re.escape(code), "", name_text).replace("|", " ").strip()
            expanded_rows.append({
                "Code": code,
                "Name": cleaned_name,
                "Page Range": page_range,
                "Description": "",
                "Relevant Pages": relevant_pages
            })

        # Expand Description-based codes
        for code, _ in desc_chunks:
            cleaned_desc = re.sub(re.escape(code), "", description_text).replace("|", " ").strip()
            expanded_rows.append({
                "Code": code,
                "Name": "",
                "Page Range": page_range,
                "Description": cleaned_desc,
                "Relevant Pages": relevant_pages
            })

    return pd.DataFrame(expanded_rows, columns=["Code", "Name", "Page Range", "Description", "Relevant Pages"])

def split_rows_by_pipe_alignment_with_codes(df):
    """Split rows where multiple ESRS codes exist in 'Name' and align with 'Description' using pipe index."""
    new_rows = []

    for _, row in df.iterrows():
        name_text = str(row.get("Name", "") or "")
        desc_text = str(row.get("Description", "") or "")
        code = row.get("Code", "")
        page_range = row.get("Page Range", "")
        relevant_pages = row.get("Relevant Pages", "")

        # Split by pipe and strip whitespace
        name_parts = [n.strip() for n in re.split(r'\s*\|\s*', name_text)]
        desc_parts = [d.strip() for d in re.split(r'\s*\|\s*', desc_text)]

        # Extract code chunks from name parts
        for i, name_chunk in enumerate(name_parts):
            matched_codes = extract_code_chunks(name_chunk)

            # Try to get matching description part
            description_chunk = desc_parts[i] if i < len(desc_parts) else ""

            if matched_codes:
                for code_found, _ in matched_codes:
                    # Remove code from name_chunk with surrounding whitespace and optional "ESRS"
                    name_cleaned = re.sub(
                        rf'\b(ESRS\s*)?{re.escape(code_found)}\b', '', name_chunk, flags=re.IGNORECASE
                    ).strip()
                    new_rows.append({
                        "Code": code_found.strip(),
                        "Name": name_cleaned,
                        "Page Range": page_range,
                        "Description": description_chunk,
                        "Relevant Pages": relevant_pages
                    })
            else:
                new_rows.append({
                    "Code": code.strip(),
                    "Name": name_chunk,
                    "Page Range": page_range,
                    "Description": description_chunk,
                    "Relevant Pages": relevant_pages
                })

    return pd.DataFrame(new_rows, columns=["Code", "Name", "Page Range", "Description", "Relevant Pages"])

# ✅ Function: preserve content + separate valid codes + filter out header/title rows
def detect_and_correct_misplaced_codes(df):
    new_rows = []

    code_pattern = re.compile(
        r'\b(?:ESRS\s*)?((?:GOV|[EGS])\d*(?:[-.]\d+)+(?:[a-zA-Z])?)\b',
        re.IGNORECASE
    )

    for _, row in df.iterrows():
        code, name, page_range, description = row["Code"], row["Name"], row["Page Range"], row["Description"]

        if not code or pd.isna(code) or str(code).strip() == "":
            matches = code_pattern.findall(name)
            if matches:
                detected_code = matches[0].strip()
                if not re.fullmatch(r'\d+(\.\d+)?', detected_code):  # avoid floats like 1.1
                    code = detected_code
                    name = name.replace(detected_code, '').strip()

        new_rows.append([code, name, page_range, description])

    return pd.DataFrame(new_rows, columns=["Code", "Name", "Page Range", "Description"])

def clean_relevant_pages_from_range(row):
    rel_pages = {int(p) for p in str(row["Relevant Pages"]).split(",") if p.strip().isdigit()}
    page_range_nums = {int(p) for p in re.findall(r'\d{2,4}', str(row["Page Range"]))}

    cleaned = sorted(rel_pages - page_range_nums)
    return ",".join(str(p) for p in cleaned)


def extract_relevant_pages(description: str, name: str = "", page_range: str = "") -> str:
    import re

    def extract_from_text(text):
        text = text.lower()
        pages = set()
        parts = re.split(r'\s*\|\s*', text)

        for part in parts:
            if re.search(r'\b(paragraph|section)\s+\d+', part):
                continue

            part = re.sub(r'\b(?:\d{1,2}[-/.]){2}\d{2,4}\b', '', part)
            part = re.sub(r'\b(january|february|march|april|may|june|july|august|'
                          r'september|october|november|december)\b', '', part)
            part = re.sub(r'\b[a-z]?\d+(?:\.\d+){1,3}[a-z]?\b', '', part)

            for start, end in re.findall(r'(?<!\d)(\d{2,4})\s*[-–—]\s*(\d{2,4})(?!\d)', part):
                s, e = int(start), int(end)
                if s < e and 10 <= s <= 1500 and 10 <= e <= 1500:
                    pages.update(range(s, e + 1))

            for match in re.findall(r'\b(\d{2,4})\b', part):
                n = int(match)
                if 10 <= n <= 1500:
                    pages.add(n)

            if re.search(r'(\d{2,4})\s*\|$', part):
                n = int(re.search(r'(\d{2,4})\s*\|$', part).group(1))
                if 10 <= n <= 1500:
                    pages.add(n)

            tokens = part.strip().split()
            if tokens and tokens[-1].isdigit():
                n = int(tokens[-1])
                if 10 <= n <= 1500:
                    pages.add(n)

        return pages

    # ✅ Extract numbers from name and description
    name_pages = extract_from_text(name if isinstance(name, str) else "")
    desc_pages = extract_from_text(description if isinstance(description, str) else "")
    all_pages = name_pages.union(desc_pages)

    # ✅ Parse page_range string → set of ints
    excluded = set()
    if isinstance(page_range, str):
        for val in re.findall(r'\d{2,4}', page_range):
            try:
                n = int(val)
                if 10 <= n <= 1500:
                    excluded.add(n)
            except:
                continue

    # ✅ Remove duplicates and page range values
    final_pages = sorted(p for p in all_pages if p not in excluded)
    return ",".join(str(p) for p in final_pages)
    

def clean_extracted_content(df):
    """Clean DataFrame after code extraction"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
        
    df = df.copy()
    for idx, row in df.iterrows():
        for col in ['Name', 'Description']:
            if col not in df.columns:
                continue
                
            text = str(row[col]) if pd.notna(row[col]) else ""
            try:
                for code, _ in extract_code_chunks(text):
                    text = re.sub(rf'(\|?\s*{re.escape(code)}\s*\|?)', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                df.at[idx, col] = text
            except Exception as e:
                print(f"Error cleaning row {idx} column {col}: {str(e)}")
                df.at[idx, col] = text.strip()
    
    return df

def full_content_preserving_standardize(filepath):
    """Standardize a single CSV file with error handling"""
    try:
        input_df = pd.read_csv(filepath)
        input_df.columns = input_df.columns.str.strip()  # ✅ Strip column names

# ✅ Extract page number from filename or folder
        file_path = Path(filepath)
        page_match = re.search(r'page[_\-]?0*([1-9]\d{0,3})', str(file_path.name))  # Try from filename
        if not page_match:
            page_match = re.search(r'page[_\-]?0*([1-9]\d{0,3})', str(file_path.parent.name))  # Fallback to folder

        page_str = page_match.group(1) if page_match else ''

# ✅ Add or overwrite Page Range and Relevant Pages columns
        input_df["Page Range"] = page_str
        input_df["Relevant Pages"] = page_str

# Add Page Range column if missing
        if "Page Range" not in input_df.columns:
            input_df["Page Range"] = page_str
    except Exception as e:
        return None, f"Error reading {filepath}: {e}"

    # Initial cleaning and standardization
    input_df.dropna(axis=1, how='all', inplace=True)
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = input_df[col].fillna('')

    standardized_rows = []
    for _, row in input_df.iterrows():
        cells = [str(cell).strip() for cell in row if str(cell).strip()]
        if not cells:
            continue

        code = name = description = ""
        page_range = row['Page Range'] if 'Page Range' in row else ''
        
        # Pattern 1: Combined "ESRS X-Y — Title"
        match_combined = combined_code_pattern.match(cells[0])
        if match_combined:
            code = match_combined.group(1).strip()
            name = match_combined.group(2).strip()
            description = " ".join(cells[1:]) if len(cells) > 1 else ""
        
        # Pattern 2: Code in first cell
        elif esrs_code_pattern.match(cells[0]):
            code = cells[0]
            name = cells[1] if len(cells) > 1 else ""
            description = " ".join(cells[2:]) if len(cells) > 2 else ""
        
        # Pattern 3: Extended
        elif extended_code_pattern.match(cells[0]):
            code = cells[0]
            name = cells[1] if len(cells) > 1 else ""
            description = " ".join(cells[2:]) if len(cells) > 2 else ""
        
        # Fallback
        else:
            code, name = split_code_and_name(cells[0])
            description = " ".join(cells[1:]) if len(cells) > 1 else ""

        standardized_rows.append([code, name, page_range, description])

    try:
        df_cleaned = pd.DataFrame(
            standardized_rows,
            columns=["Code", "Name", "Page Range", "Description"]
        )
        
        # Apply processing pipeline
        processed_df = process_dataframe(df_cleaned)
        processed_df = detect_and_correct_misplaced_codes(processed_df)

# 🔁 NEW STEP: First split by aligned pipes with code pairing
        processed_df = split_rows_by_pipe_alignment_with_codes(processed_df)

# 🔁 THEN: Fallback to embedded code splitting if needed
        processed_df = split_rows_on_any_embedded_esrs_code(processed_df)

        processed_df["Relevant Pages"] = page_str
        
        # Final grouping
        final_df = processed_df.groupby('Code').agg({
            'Name': lambda x: " | ".join(x.dropna().unique()),
            'Page Range': 'first',
            'Description': lambda x: " | ".join(x.dropna().unique()),
            'Relevant Pages': lambda x: ",".join(sorted(
                {p for p in ",".join(x).split(",") if p.strip()},
                key=lambda p: int(p) if p.isdigit() else float('inf')
            ))
        }).reset_index()
        
        return final_df, None
        
    except Exception as e:
        return None, f"Error processing {filepath}: {str(e)}"

def extract_section_reference(name: str, description: str) -> str:
    """
    Extracts section/paragraph references from name & description with logic:
    - Allows combinations starting with a number (e.g., 4.2a, 3-1, 2.1.1) but never starting with 0
    - Standalone numbers (e.g., '22') are only extracted if preceded by 'section', 'paragraph', or 'chapter'
    - Captures trailing titles only if the first word starts with a capital letter
    - Discards anything starting with a letter (e.g., 'E1-4')
    - Skips false matches like numbers inside 'ESRS-2'
    - Skips anything ending in .0, -0, or (0)
    - Skips combinations that look like page ranges (e.g. 127-128, 101.0) if values fall in page range range
    - Ensures sections are not captured if followed directly by likely page numbers
    - Final results are comma-separated (not pipe)
    """

    import re

    def find_phrases(text):
        if not isinstance(text, str):
            return []

        text = re.sub(r'[‐‒–—]', '-', text).strip()

        pattern = re.compile(
            r'(?<![\w-])'                                   # Not part of another word or dash before
            r'(?:(?P<prefix>paragraph|section|chapter)\s+)?'  # Optional prefix: paragraph, section, or chapter
            r'(?P<ref>[1-9]\d{0,3}(?:[-.]\d+){0,6}(?:\([a-z]\))?)'  # Must start with non-zero digit
            r'(?:\s+(?P<tail>[A-Z][^\d\|\n\r]*))?'      # Optional phrase (must start with capital)
            r'(?=\s+(?!\d{1,4}(?![.-]))|\||\n|$)',        # Stop unless followed by clear page number
            flags=re.IGNORECASE
        )

        matches = []
        for match in pattern.finditer(text):
            prefix = match.group("prefix") or ""
            ref = match.group("ref").strip()
            tail = (match.group("tail") or '').strip()

            if not ref[0].isdigit() or ref[0] == '0':
                continue

            if re.fullmatch(r'\d{1,4}', ref) and not prefix:
                continue

            if ref.isdigit() and int(ref) > 1000:
                continue

            if re.search(r'(\.0|-0|\(0\))$', ref):
                continue

            if re.fullmatch(r'\d{2,4}[-.]\d{1,4}', ref):
                try:
                    parts = [int(x) for x in re.split(r'[-.]', ref)]
                    if all(10 <= x <= 1500 for x in parts):
                        continue
                except:
                    pass

            combined = f"{prefix} {ref}".strip()
            if tail:
                combined += f" {tail}"
            matches.append(combined.strip())

        return matches

    refs = find_phrases(name) + find_phrases(description)
    return ", ".join(dict.fromkeys(refs))  # unique & comma-separated