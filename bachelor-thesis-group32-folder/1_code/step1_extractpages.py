import sys
from pathlib import Path
sys.path.append(".")
from libraries.imports import *

#THIS .PY FILE HAVE ONLY FUNCTION AND VARIABLE DEFINITIONS

#DR pattern that should be met under DR column
DR_pattern = re.compile(
    r"\b(?:ESRS\s*)?(?:[EGS](?:\d+(?:[-–]\d+)?)?|GOV|SBM|BP|SMB|IRO|MDR)"
    r"(?:[\s\-–]+(?:ESRS\s*)?(?:[EGS](?:\d+(?:[-–]\d+)?)?|GOV|SBM|BP|SMB|IRO|MDR))*"
    r"\b",
    re.IGNORECASE
)

#in addition to DR list there might be entity specific DR list
ENTITY_pattern = re.compile(
    r"\b(entity[\s\-]*specific|entity[\s\-]*related[\s\-]*disclosure|"
    r"custom[\s\-]*disclosure|entity[\s\-]*level\s+esrs)\b",
    re.IGNORECASE
)

#keywords to find the relavant pages where the DR table is located
keywords = ["disclosure requirement", "location", "standard section", 
           "reference table", "content index", "ESRS indices", "reference in the report", "index of", "section", "page number", "reference to", "list of ESRS disclosure requirements"]

#the DR table is usually followed by EU legislation table hence we need to distinguish then
EU_pattern = re.compile(
    r"\b("
    r"eu\s+legislation|"
    r"e\.u\.\s+legislation|"
    r"eu\s+taxonomy|"
    r"eu\s+taxonomy\s+regulation|"
    r"eu\s+regulation|"
    r"sfdr|"
    r"sfdr\s+reference|"
    r"pillar\s+3\s+reference|"
    r"benchmark\s+regulation\s+reference|"
    r"directive\s+\d+/\d+/EC|"
    r"regulation\s+\(EU\)\s+\d+/\d+|"
    r"due\s+diligence|"
    r"elements\s+of\s+due\s+diligence|"
    r"law\s+11/2018|"
    r"material\s+impacts|"
    r"financial\s+materiality"
    r")\b",
    re.IGNORECASE
)

#%%
def normalize_whitespace(text):
    """Removes multiple whitespaces and lease only one ' ' whitespace"""
    return re.sub(r'\s+', ' ', text).strip()

def normalize_unicode(text):
    """Uses Normalization From Compatibility Composition to replace non-standard unicode characters (the ones that might be written in LaTeX or some special characters)"""
    return unicodedata.normalize("NFKC", text)

def looks_like_table(text):
    lines = text.splitlines()
    table_like_lines = sum(1 for line in lines if len(re.findall(r'\s{2,}', line)) >= 1)
    return table_like_lines >= 3

def extract_text_and_score_pages(pdf_path,esrs_count_weight=1.125,unique_esrs_weight=1.125,keyword_count_weight=3.75,has_table_structure_weight=0.075,eu_penalty_weight=3.75):
    doc = fitz.open(pdf_path)
    results = []

    for page_num, page in enumerate(doc):
        text = normalize_unicode(page.get_text())
        lowered = text.lower()
        lowered = normalize_whitespace(lowered)

        # Count ESRS codes and entity specific codes
        dr_matches = DR_pattern.findall(lowered)
        entity_matches = ENTITY_pattern.findall(lowered)
        all_dr_matches = dr_matches+entity_matches
        esrs_count = len(all_dr_matches)
        unique_esrs = set(all_dr_matches)


        # Count keyword hits
        keyword_count = sum(kw in lowered for kw in keywords)

        # Detect table structure
        has_table_structure = looks_like_table(text)

        #count eu legislation mentionings
        eu_hits = bool(EU_pattern.search(lowered))
        eu_penalty = 1 if eu_hits else 0  

        #total scoring system. the most weight is to the keyword and eu legislation penalty
        score = (esrs_count*esrs_count_weight +len(unique_esrs) * unique_esrs_weight + keyword_count * keyword_count_weight + (1 if has_table_structure else 0)*has_table_structure_weight - eu_penalty*eu_penalty_weight)

        results.append({
        "page_num": page_num,
        "total_score": score,
        "keyword_hits": keyword_count,
        "unique_esrs": len(unique_esrs),
        "total_esrs": esrs_count,
        "has_table": has_table_structure,
        "eu_legislation_hit": eu_hits,
        "eu_penalty": eu_penalty
    })

    #Remove early pages (e.g. first 25) (less likely too appear that early in the text)
    df = pd.DataFrame(results)
    df_filtered = df[df["page_num"] > 25].copy()
    df_sorted = df_filtered.sort_values("total_score", ascending=False).reset_index(drop=True)
    return df_sorted


#%%
def get_expanded_page_range(df, pdf_path, esrs_thresh=5, unique_thresh=4):
    """Takes the top one page candidate and looks through its neighbours to find a range of table pages.
    Note: the logic is that the page range should cover the tables beginning and the end,
    hence there might be +/- 1 not relevant page but that ensures the extraction of the whole table later."""
    
    df = df.copy()
    numeric_cols = ['page_num', 'total_score', 'keyword_hits', 'unique_esrs', 'total_esrs']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    top_page = int(df.iloc[0]['page_num'])
    assessed_pages = []
    final_pages = [top_page]

    doc = fitz.open(pdf_path)

    def assess_page(page_num):
        if page_num not in df['page_num'].values:
            return None
        row = df[df['page_num'] == page_num].iloc[0]

        text = doc[page_num].get_text().lower()
        eu_hits = EU_pattern.findall(text)
        eu_penalty = len(eu_hits)

        meets_criteria = (
            (row['total_esrs'] >= esrs_thresh or row['unique_esrs'] >= unique_thresh)
        )

        return {
            'page': page_num,
            'total_score': row['total_score'],
            'total_esrs': row['total_esrs'],
            'unique_esrs': row['unique_esrs'],
            'eu_penalty': eu_penalty,
            'meets_criteria': meets_criteria,
            'in_final_range': False
        }

    # Backward neighbours
    high_eu_seen = False
    for offset in range(1, 7):
        i = top_page - offset
        if i < 0:
            break
        result = assess_page(i)
        if result:
            assessed_pages.append(result)
            row = result

            if not row['meets_criteria']:
                break

            final_pages.insert(0, i)
            result['in_final_range'] = True

            if high_eu_seen:
                if row['eu_penalty'] > 9:
                    break
            else:
                if row['eu_penalty'] > 20:
                    break
                high_eu_seen = True
        else:
            break

    # Forward neighbours
    high_eu_seen = False
    for offset in range(1, 7):
        i = top_page + offset
        if i >= len(doc):
            break
        result = assess_page(i)
        if result:
            assessed_pages.append(result)
            row = result

            if not row['meets_criteria']:
                break

            final_pages.append(i)  
            result['in_final_range'] = True

            if high_eu_seen:
                if row['eu_penalty'] > 9:
                    break  
            else:
                if row['eu_penalty'] > 20:
                    break  
                high_eu_seen = True
        else:
            break

    doc.close()

    assessment_df = pd.DataFrame(assessed_pages).sort_values("page")
    return {
        "final_page_range": sorted(final_pages),
        "assessment_details": assessment_df
    }


#created a function that crops the original pdf to the extracted page range and saves it under the same pdf name + _C (for crop)
def crop_pdf_to_range_and_preview(pdf_path, page_range, preview_lines=5):
    """
    Crops a PDF to the specified page range and saves it inside 0_data/cropped_pdf/
    relative to where the PDF already is.
    """
    doc = fitz.open(str(pdf_path))
    cropped_doc = fitz.open()
    total_pages = len(doc)

    print(f" PDF has {total_pages} pages total.")
    print(f" Attempting to crop pages: {page_range}")

    for page_num in page_range:
        if 0 <= page_num < total_pages:
            try:
                page_text = doc[page_num].get_text().strip().split("\n")
                print(f"\n Page {page_num + 1} (index {page_num}) preview:")
                print("\n".join(page_text[:preview_lines]))
                print("-" * 60)

                cropped_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            except Exception as e:
                print(f" Error inserting page {page_num}: {e}")
        else:
            print(f" Page number {page_num} is out of bounds (PDF has {total_pages} pages).")

    if len(cropped_doc) == 0:
        print("No pages were added. Cropped PDF will not be saved.")
        doc.close()
        return None

    pdf_path = Path(pdf_path)
    output_folder = Path(__file__).resolve().parents[1] / "0_data" / "cropped_pdf"
    os.makedirs(output_folder, exist_ok=True)

    filename = pdf_path.stem + "_c.pdf"
    output_path = output_folder / filename

    cropped_doc.save(str(output_path))
    cropped_doc.close()
    doc.close()

    print(f"\n Cropped PDF saved as: {output_path}")
    return output_path