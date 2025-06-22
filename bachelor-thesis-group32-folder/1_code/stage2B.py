# Stage2B.py: Non‑interactive ESRS Table Filtering & Selection
# --------------------------------------------------------------
# Refactored for notebook/CI use—no prompts.
# Usage: import main or run; CLI entrypoint also available.

import os
import re
import pandas as pd
from itertools import groupby
import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Patterns and helper definitions

DR_pattern = re.compile(
    r"\b(?:ESRS\s*)?(?:[EGS](?:\d+(?:[-–]\d+)?)?|GOV|SBM|BP|SMB|IRO|MDR)"
    r"(?:[\s\-–]+(?:ESRS\s*)?(?:[EGS](?:\d+(?:[-–]\d+)?)?|GOV|SBM|BP|SMB|IRO|MDR))*"
    r"\b",
    re.IGNORECASE
)
EU_pattern = re.compile(
    r"\b(eu\s+legislation|eu\s+taxonomy|sfdr|directive\s+\d+/\d+/EC|"
    r"regulation\s+\(EU\)\s+\d+/\d+|due\s+diligence)\b",
    re.IGNORECASE
)
filter_name_patterns = [
    re.compile(r"\bdisclosures?\b", re.IGNORECASE),
    re.compile(r"\bdisclosure\s+requirement\b", re.IGNORECASE),
    re.compile(r"\bESRS\s+disclosure\b", re.IGNORECASE),
    re.compile(r"\bchange\s+form\b", re.IGNORECASE)
]
explanation_pattern = re.compile(r"\bESRS.*explanation\b", re.IGNORECASE)
exclude_terms = ["datapoint","data point","datapoints","data points"]
stage1_keywords = [
    "disclosure requirement","location","standard section",
    "reference table","content index","ESRS indices",
    "reference in the report","index of","section",
    "page number","reference to","list of ESRS disclosure requirements"
]
MIN_DR_TOTAL, MIN_DR_COL_COUNT, MIN_DR_COL_RATIO = 3, 3, 0.4

# ──────────────────────────────────────────────────────────────────────────────
# Heuristic helpers

def is_paragraph(df):
    return len(df.columns) <= 1

def looks_like_table(text):
    lines = text.splitlines()
    tbl  = sum(bool(re.search(r"\s{2,}", l)) for l in lines)
    long = sum(len(l) > 10 for l in lines)
    return tbl >= 3 and long > tbl * 0.5

def has_unnamed_header(df):
    if df.columns.to_series().str.contains(r"^Unnamed", case=False).any(): return True
    return df.iloc[0].astype(str).str.contains(r"^Unnamed", case=False).any()

# ──────────────────────────────────────────────────────────────────────────────
# Core filtering function

def filter_table(df):
    # full text for density checks
    text_all = " ".join(df.astype(str).apply(lambda r: " ".join(r), axis=1)).lower()
    # header slice: columns + first four rows (idx 0–3)
    header = df.head(4).astype(str).apply(lambda r: " ".join(r), axis=1)
    text_header = " ".join(df.columns.astype(str).tolist() + header.tolist()).lower()

    # 1) header-slice bans
    if any(term in text_header for term in exclude_terms): return False
    if EU_pattern.search(text_header): return False
    # 2) drop pure narratives
    if is_paragraph(df): return False
    # 3) dense-column rule
    for col in df.columns:
        hits = df[col].astype(str).apply(lambda v: bool(DR_pattern.search(v))).sum()
        if hits >= MIN_DR_COL_COUNT and hits / max(len(df),1) >= MIN_DR_COL_RATIO:
            return True
    # 4) overall code count fallback
    if len(DR_pattern.findall(text_all)) >= MIN_DR_TOTAL:
        return True
    # 5) header/name cues if codes exist
    total_hits = len(DR_pattern.findall(text_all))
    if total_hits >= MIN_DR_TOTAL:
        if any(p.search(str(c)) for p in filter_name_patterns for c in df.columns): return True
        first_row = [str(v) for v in df.iloc[0]]
        if any(p.search(v) for p in filter_name_patterns for v in first_row): return True
        if explanation_pattern.search(text_all): return True
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Scoring logic

def compute_score(df, page_num, engine, fname):
    if is_paragraph(df): return -999
    score = 0
    cols  = list(df.columns)
    generic = sum(c.lower().startswith("unnamed") or not re.search(r"[A-Za-z]",c) for c in cols)
    score += (len(cols)-generic) - 2*generic
    dr_text = " ".join(df[c].astype(str).str.cat(sep=" ") for c in cols if DR_pattern.search(" ".join(df[c].astype(str)))).lower()
    hits = DR_pattern.findall(dr_text)
    score += 1.5*(len(hits)+len(set(hits)))
    for kw in stage1_keywords:
        if kw in dr_text: score+=5
    pg = [c for c in cols if re.search(r"\bpage(?:s)?\b",c,re.I)]
    score += 8*len(pg)
    if engine=="tabula": score+=2
    if pg:
        nums=[int(m.group(1)) for m in (re.match(r"^\s*(\d+)",str(v)) for v in df[pg[0]]) if m]
        if nums and sum(n==page_num for n in nums)/len(nums)>0.6: score+=1
    score+=0.1 if looks_like_table(dr_text) else 0
    score-=15 if EU_pattern.search(dr_text) else 0
    score+=len(df)*0.01
    return score

# ──────────────────────────────────────────────────────────────────────────────
# Non-interactive runner

def run(input_dir, output_dir, threshold=10.0):
    os.makedirs(output_dir, exist_ok=True)
    reports = [os.path.join(input_dir,d) for d in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir,d))]
    candidates=[]
    regex = re.compile(r"page_(\d+).*?(camelot|tabula).*?\.csv$",re.I)
    for rep in reports:
        for fn in sorted(os.listdir(rep)):
            if not fn.lower().endswith('.csv'): continue
            m=regex.search(fn)
            if not m: continue
            p,e=int(m.group(1)),m.group(2).lower()
            df=pd.read_csv(os.path.join(rep,fn))
            if not filter_table(df): continue
            sc=compute_score(df,p,e,fn)
            if sc<threshold and has_unnamed_header(df): continue
            candidates.append((p,fn,e,sc,df,rep))
    if not candidates:
        print("No tables passed the filters.")
        return
    best={}
    for p,group in groupby(sorted(candidates,key=lambda x:(x[0],-x[3])), key=lambda x:x[0]):
        lst=list(group)
        clean=[t for t in lst if not has_unnamed_header(t[4])]
        pool=clean if clean else lst
        best[p]=max(pool,key=lambda x:x[3])
    for p,(_,fn,_,sc,df,rep) in sorted(best.items()):
        sub=os.path.join(output_dir,os.path.basename(rep))
        os.makedirs(sub,exist_ok=True)
        df.to_csv(os.path.join(sub,fn),index=False)
        print(f"Page {p}: saved {fn}, score={sc:.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint

if __name__=='__main__':
    parser=argparse.ArgumentParser(desc='Stage2B filter')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--threshold',type=float,default=10.0)
    a=parser.parse_args()
    run(a.input_dir,a.output_dir,a.threshold)
