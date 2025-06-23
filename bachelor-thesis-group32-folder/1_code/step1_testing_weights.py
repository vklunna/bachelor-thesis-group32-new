from step1_extractpages import extract_text_and_score_pages, get_expanded_page_range

def safe_int(p):
    try:
        return int(p)
    except:
        return None

def evaluate_range_match(pred_pages, true_start, true_end):
    cleaned = [safe_int(p) for p in pred_pages if safe_int(p) is not None]
    if not cleaned:
        return {"perfect_match": False, "acceptable_match": False}

    pred_start, pred_end = cleaned[0], cleaned[-1]
    true_range = set(range(true_start, true_end + 1))
    pred_range = set(cleaned)

    perfect = pred_start == true_start and pred_end == true_end
    acceptable = true_range.issubset(pred_range)

    return {"perfect_match": perfect, "acceptable_match": acceptable}

def rescore_df(df, weights):
    df = df.copy()
    df["total_score"] = (
        df["total_esrs"] * weights["esrs"] +
        df["unique_esrs"] * weights["unique"] +
        df["keyword_hits"] * weights["keyword"] +
        df["has_table"].astype(int) * weights["table"] -
        df["eu_penalty"] * weights["eu"]
    )
    return df.sort_values("total_score", ascending=False).reset_index(drop=True)

def run_weight_grid_search(label_file_path):
    import pandas as pd
    from pathlib import Path
    from itertools import product
    from tqdm import tqdm
    from step1_extractpages import extract_text_and_score_pages, get_expanded_page_range
    import os

    labels_df = pd.read_excel(label_file_path)

    weight_grid = [
        {"esrs": ew, "unique": uw, "keyword": kw, "table": tw, "eu": euw}
        for ew, uw, kw, tw, euw in product(
            [1.125, 1.5, 1.875],     # ESRS count weight
            [1.125, 1.5, 1.875],     # Unique ESRS weight
            [3.75, 5, 6.25],         # Keyword weight
            [0.075, 0.1, 0.125],     # Table structure weight
            [3.75, 5, 6.25]          # EU penalty weight
        )
    ]

    # === Preprocess all PDFs once ===
    pdf_cache = {}
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        raw_path = str(row["pdf_path"]).strip()
        full_path = str(Path("../0_data") / raw_path)
        if not os.path.exists(full_path):
            print(f"Missing: {full_path}")
            continue
        try:
            df = extract_text_and_score_pages(full_path)
            pdf_cache[full_path] = df
        except Exception as e:
            print(f"Failed to parse {full_path}: {e}")
            continue

    # === Grid Search Evaluation ===
    results = []
    for weights in tqdm(weight_grid):
        perfect_hits, acceptable_hits, total = 0, 0, 0

        for _, row in labels_df.iterrows():
            raw_path = str(row["pdf_path"]).strip()
            full_path = str(Path("../0_data") / raw_path)
            if full_path not in pdf_cache:
                continue

            try:
                true_start = int(row["start"])
                true_end = int(row["end"])

                rescored_df = rescore_df(pdf_cache[full_path], weights)
                pred_result = get_expanded_page_range(rescored_df, full_path)
                pred_pages = pred_result["final_page_range"]

                print(f"Evaluating {os.path.basename(full_path)}: pred={pred_pages}, true=({true_start}-{true_end})")

                match = evaluate_range_match(pred_pages, true_start, true_end)
                perfect_hits += int(match["perfect_match"])
                acceptable_hits += int(match["acceptable_match"])
                total += 1

            except Exception as e:
                print(f"Error evaluating {full_path}: {e}")
                continue

        if total > 0:
            results.append({
                **weights,
                "perfect_match_rate": perfect_hits / total,
                "acceptable_match_rate": acceptable_hits / total,
                "total_evaluated": total
            })

    summary_df = pd.DataFrame(results)

    if summary_df.empty:
        print("No successful evaluations.")
    else:
        summary_df.sort_values(by=["perfect_match_rate", "acceptable_match_rate"], ascending=[False, False], inplace=True)

    return summary_df, summary_df.iloc[0]

#evaluating new weights and comparing them to the original ones so check for differences
def test_weights_on_pdfs(label_file_path, best_weights):
    print("Running test_weights_on_pdfs from:", __file__)
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    from step1_extractpages import extract_text_and_score_pages, get_expanded_page_range

    labels_df = pd.read_excel(label_file_path)
    extracted_ranges = []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        raw_path = str(row["pdf_path"]).strip()
        full_path = Path("../0_data") / raw_path

        if not full_path.exists():
            print(f"File not found: {full_path}")
            continue

        try:
            scored_df = extract_text_and_score_pages(
                full_path,
                esrs_count_weight=best_weights["esrs"],
                unique_esrs_weight=best_weights["unique"],
                keyword_count_weight=best_weights["keyword"],
                has_table_structure_weight=best_weights["table"],
                eu_penalty_weight=best_weights["eu"]
            )
            result = get_expanded_page_range(scored_df, full_path)
            page_range = result["final_page_range"]

            extracted_ranges.append({
                "pdf_path": raw_path,
                "extracted_start": page_range[0] if page_range else None,
                "extracted_end": page_range[-1] if page_range else None,
                "candidate_range": page_range
            })

        except Exception as e:
            print(f"Error in {full_path.name}: {e}")
            continue

    return pd.DataFrame(extracted_ranges)