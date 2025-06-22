import pandas as pd
import os
from tqdm import tqdm
import json
import io # For OCR text parsing

# Attempt to import new libraries for advanced table extraction
try:
    import camelot
except ImportError:
    print("Warning: camelot-py library not found. Install: pip install \"camelot-py[cv]\" + Ghostscript")
    camelot = None
try:
    import tabula
except ImportError:
    print("Warning: tabula-py library not found. Install: pip install tabula-py + Java JDK")
    tabula = None
try:
    import pdfplumber
except ImportError:
    print("Warning: pdfplumber library not found. Install: pip install pdfplumber")
    pdfplumber = None
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Warning: PyMuPDF (fitz) library not found. Install: pip install PyMuPDF")
    fitz = None
try:
    import pytesseract # For OCR
    # For Windows, if Tesseract is not in PATH, uncomment and set the path below:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    print("Warning: pytesseract library not found. Install: pip install pytesseract. OCR functionality will be disabled.")
    print("Ensure Tesseract OCR engine is also installed on your system: https://github.com/tesseract-ocr/tessdoc")
    pytesseract = None

# --- Helper Functions (is_good_table, is_reportable_table, make_unique_headers, ocr_text_to_dataframe) ---
# These functions are assumed to be the same as in your original script.
# Make sure ocr_text_to_dataframe is defined as in your second script.

def is_good_table(df, min_rows=1, min_cols=1):
    """Checks if a DataFrame is a 'good' table (not None, not empty, has content)."""
    if df is None or df.empty:
        return False
    if df.shape[0] < min_rows or df.shape[1] < min_cols:
        return False
    for r_idx in range(min(3, df.shape[0])):
        for c_idx in range(df.shape[1]):
            cell_value = df.iat[r_idx, c_idx]
            if cell_value is not None and str(cell_value).strip() != "":
                return True
    return False

def is_reportable_table(df, method_name="Unknown Method", table_index=-1, max_avg_cell_len=200, min_cols_for_data=2, max_cols_for_paragraph=2):
    if not is_good_table(df, min_rows=1, min_cols=1): # Use a basic min_rows=1, min_cols=1 here
        print(f"      FILTER ({method_name} T{table_index}): FAILED basic is_good_table check.")
        return False
    rows, cols = df.shape
    total_chars = 0
    num_cells_with_content = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            cell_val_str = str(df.iat[r_idx, c_idx]).strip()
            if cell_val_str and cell_val_str.lower() != 'none':
                total_chars += len(cell_val_str)
                num_cells_with_content += 1
    avg_cell_len = 0
    if num_cells_with_content > 0:
        avg_cell_len = total_chars / num_cells_with_content
    if cols <= max_cols_for_paragraph and avg_cell_len > max_avg_cell_len:
        print(f"      FILTER ({method_name} T{table_index}): FAILED reportable check: {cols} col(s) with avg cell length {avg_cell_len:.0f} > {max_avg_cell_len} (likely paragraph block).")
        return False
    if rows <= 2 and cols > 10 and avg_cell_len < 15:
        print(f"      FILTER ({method_name} T{table_index}): FAILED reportable check: {rows} row(s), {cols} cols, short avg cell length {avg_cell_len:.0f} (potential header/footer block).")
        return False
    if rows == 1 and cols == 1 and num_cells_with_content == 1 and total_chars > max_avg_cell_len * 2 :
        print(f"      FILTER ({method_name} T{table_index}): FAILED reportable check: Single very long cell detected ({total_chars} chars).")
        return False
    print(f"      PASSED Reportable Table Check ({method_name} T{table_index}): {rows}r x {cols}c, avg_len={avg_cell_len:.0f}")
    return True

def make_unique_headers(header_list):
    processed_headers = []
    header_occurrences = {}
    for header_val_orig in header_list:
        header_val = str(header_val_orig) if header_val_orig is not None else "Unnamed"
        if header_val not in header_occurrences:
            header_occurrences[header_val] = 0
            processed_headers.append(header_val)
        else:
            header_occurrences[header_val] += 1
            processed_headers.append(f"{header_val}_{header_occurrences[header_val]}")
    return processed_headers

def ocr_text_to_dataframe(ocr_text): # From your second script
    if not ocr_text or ocr_text.isspace():
        return pd.DataFrame()
    lines = [line.strip() for line in ocr_text.strip().split('\n') if line.strip()]
    data = []
    for line in lines:
        # Naive split by multiple spaces; consider more robust CSV-like parsing if needed
        cells = [cell.strip() for cell in line.split('  ') if cell.strip()] 
        if cells:
            data.append(cells)
    if not data:
        return pd.DataFrame()
    try:
        max_cols = max(len(row) for row in data) if data else 0
        padded_data = [row + [None] * (max_cols - len(row)) for row in data]
        df = pd.DataFrame(padded_data)
        # Attempt to promote first row to header if it looks like one
        if not df.empty and len(df) > 1:
            first_row_values = df.iloc[0].tolist()
            # Heuristic: all strings, not all numbers, more than one unique value
            is_header_like = all(isinstance(x, str) for x in first_row_values if x is not None) and \
                             not all(str(x).replace('.', '', 1).isdigit() for x in first_row_values if x is not None and str(x).strip() != "") and \
                             len(set(x for x in first_row_values if x is not None)) > 1 
            if is_header_like:
                try:
                    df.columns = make_unique_headers(first_row_values)
                    df = df.iloc[1:].reset_index(drop=True)
                except Exception as header_e:
                    print(f"        OCR DF: Could not set headers: {header_e}. Using default headers.")
                    df = pd.DataFrame(padded_data) # Recreate if header promotion fails
        elif not df.empty : # Single row data or failed multi-row header promotion
             df = pd.DataFrame(padded_data) # Ensure it's a DataFrame
    except Exception as e:
        print(f"        OCR to DataFrame conversion error: {e}")
        return pd.DataFrame()
    return df

# --- Advanced Table Extraction Function ---
def extract_tables_advanced(pdf_path, page_numbers_0_indexed):
    extracted_data_final = []
    pages_to_process = sorted(list(set(page_numbers_0_indexed)))
    
    DEBUG_PYMUPDF_FOR_PAGE_0_IDX = -1 
    OCR_METHOD_ENABLED = True 

    # Define parameters for table extraction methods
    camelot_parameters_lattice = {'flavor': 'lattice', 'suppress_stdout': True, 'line_scale': 30}
    camelot_parameters_stream = {'flavor': 'stream', 'suppress_stdout': True, 'edge_tol': 100, 'row_tol': 10}
    # tabula_parameters: pages will be set per page, other params are fine.
    base_tabula_parameters = {'multiple_tables': True, 'silent': True, 'guess': True}
    pymupdf_table_settings = {"strategy": "both"} # Default for PyMuPDF

    if not pages_to_process:
        print("Table Extractor: No pages specified for extraction.")
        return extracted_data_final

    print(f"\nTable Extractor: Attempting ADVANCED table extraction from {len(pages_to_process)} page(s) of '{os.path.basename(pdf_path)}': "
          f"{[p + 1 for p in pages_to_process]} (1-indexed pages)")
    print("----------------------------------------------------------------------")

    for page_num_0_idx in tqdm(pages_to_process, desc="Extracting tables (multi-method)", unit="page"):
        page_num_1_idx_str = str(page_num_0_idx + 1)
        # tables_found_on_page = [] # This was in your new snippet, replaced by selected_tables logic
        # method_used = "None" # This was in your new snippet, method name is now part of the tuple

        print(f"\nProcessing Page {page_num_1_idx_str} (0-indexed: {page_num_0_idx}) of '{os.path.basename(pdf_path)}':")

        all_method_results = []  # Store (df, method_name, confidence) from all methods
        
        # --- Method 1: Camelot (Lattice) ---
        if camelot:
            print(f"  Page {page_num_1_idx_str}: Trying Camelot (Lattice)...")
            try:
                camelot_tl = camelot.read_pdf(pdf_path, pages=page_num_1_idx_str, **camelot_parameters_lattice)
                print(f"    Camelot (Lattice) found {len(camelot_tl)} potential structure(s).")
                for i, table_report in enumerate(camelot_tl):
                    df = table_report.df
                    if is_reportable_table(df, method_name="Camelot-L", table_index=i):
                        confidence = table_report.accuracy - table_report.whitespace
                        all_method_results.append((df, "Camelot-Lattice", confidence))
                        print(f"      Added Camelot-L Table {i} (Acc:{table_report.accuracy:.1f}, WS:{table_report.whitespace:.1f}, Conf:{confidence:.1f})")
            except Exception as e:
                print(f"    Camelot (Lattice) failed: {str(e)[:200]}")

        # --- Method 2: Camelot (Stream) ---
        if camelot:
            print(f"  Page {page_num_1_idx_str}: Trying Camelot (Stream)...")
            try:
                camelot_ts = camelot.read_pdf(pdf_path, pages=page_num_1_idx_str, **camelot_parameters_stream)
                print(f"    Camelot (Stream) found {len(camelot_ts)} potential structure(s).")
                for i, table_report in enumerate(camelot_ts):
                    df = table_report.df
                    if is_reportable_table(df, method_name="Camelot-S", table_index=i):
                        confidence = table_report.accuracy - table_report.whitespace
                        all_method_results.append((df, "Camelot-Stream", confidence))
                        print(f"      Added Camelot-S Table {i} (Acc:{table_report.accuracy:.1f}, WS:{table_report.whitespace:.1f}, Conf:{confidence:.1f})")

            except Exception as e:
                print(f"    Camelot (Stream) failed: {str(e)[:200]}")

        # --- Method 3: Tabula-py ---
        if tabula:
            print(f"  Page {page_num_1_idx_str}: Trying Tabula-py...")
            try:
                current_tabula_params = base_tabula_parameters.copy()
                current_tabula_params['pages'] = page_num_1_idx_str
                tabula_dfs = tabula.read_pdf(pdf_path, **current_tabula_params)
                
                if tabula_dfs is not None: # tabula can return None if no tables
                    print(f"    Tabula-py found {len(tabula_dfs)} potential table(s).")
                    for i, df in enumerate(tabula_dfs): # Ensure iteration over potentially empty list
                        if is_reportable_table(df, method_name="Tabula", table_index=i):
                            # Estimate confidence based on table structure (more cells = higher confidence, capped)
                            confidence = min(90.0, float(df.shape[0] * df.shape[1] * 0.5)) 
                            all_method_results.append((df, "Tabula-py", confidence))
                            print(f"      Added Tabula Table {i} (Conf:{confidence:.1f})")
                else:
                    print(f"    Tabula-py found no tables.")
            except Exception as e: 
                print(f"    Tabula-py failed: {str(e)[:200]}")
                if "java" in str(e).lower() and ("runtime" in str(e).lower() or "locate" in str(e).lower()):
                    print("    Tabula-py Error: Could not find or access Java Runtime. Please ensure Java is installed and in your system PATH.")


        # --- Method 4: PyMuPDF with Redaction (Fallback as per your new logic) ---
        if fitz and not all_method_results: 
            print(f"  Page {page_num_1_idx_str}: Trying PyMuPDF with paragraph redaction (as fallback)...")
            doc = None
            try:
                doc = fitz.open(pdf_path)
                if 0 <= page_num_0_idx < len(doc):
                    page = doc.load_page(page_num_0_idx)
                    # --- PyMuPDF Redaction Logic (from original script) ---
                    table_finder_initial = page.find_tables(**pymupdf_table_settings)
                    initial_table_bboxes = []
                    for tbl_idx, table_obj_init in enumerate(table_finder_initial.tables):
                        refined_bbox_from_cells = fitz.Rect()
                        if table_obj_init.cells:
                            for cell_bbox_tuple in table_obj_init.cells:
                                if cell_bbox_tuple: refined_bbox_from_cells.include_rect(fitz.Rect(cell_bbox_tuple))
                        current_bbox_to_use = None
                        if not refined_bbox_from_cells.is_empty: current_bbox_to_use = refined_bbox_from_cells
                        elif table_obj_init.bbox: current_bbox_to_use = fitz.Rect(table_obj_init.bbox)
                        if current_bbox_to_use and not current_bbox_to_use.is_empty: initial_table_bboxes.append(current_bbox_to_use)
                    
                    paragraphs_to_redact_rects = []
                    if initial_table_bboxes:
                        text_blocks = page.get_text("blocks")
                        for block in text_blocks:
                            if block[6] == 0: # text block
                                block_rect = fitz.Rect(block[:4])
                                if not any(block_rect.intersects(table_bbox) for table_bbox in initial_table_bboxes):
                                    paragraphs_to_redact_rects.append(block_rect)
                        if paragraphs_to_redact_rects:
                            for para_rect in paragraphs_to_redact_rects: page.add_redact_annot(para_rect)
                            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                    # --- End Redaction ---
                    final_tables_finder = page.find_tables(**pymupdf_table_settings)
                    print(f"    PyMuPDF found {len(final_tables_finder.tables)} potential table structure(s) post-processing.")
                    for i, table_obj in enumerate(final_tables_finder.tables):
                        raw_table_data = table_obj.extract()
                        df = pd.DataFrame()
                        if raw_table_data and len(raw_table_data) > 0:
                            header_row = raw_table_data[0]
                            if all(h is None or str(h).strip() == "" for h in header_row):
                                num_cols = len(header_row) if header_row else (len(raw_table_data[1]) if len(raw_table_data)>1 else 0)
                                if num_cols == 0 and len(raw_table_data)>0: num_cols = 1 if any(r for r in raw_table_data) else 0
                                header = [f"Column_{j+1}" for j in range(num_cols)]; df_data = raw_table_data
                            else: header = make_unique_headers(header_row); df_data = raw_table_data[1:]
                            if df_data: df = pd.DataFrame(df_data, columns=header)
                            elif header: df = pd.DataFrame(columns=header)
                        
                        if is_reportable_table(df, method_name="PyMuPDF", table_index=i):
                            confidence = 85.0 # Assign a default confidence for PyMuPDF
                            # You could try to derive confidence from table_obj.stats if available and meaningful
                            all_method_results.append((df, "PyMuPDF-Redact", confidence))
                            print(f"      Added PyMuPDF Table {i} (Conf:{confidence:.1f})")
                else:
                    print(f"    PyMuPDF: Page index {page_num_0_idx} out of bounds for document with {len(doc)} pages.")
            except Exception as e:
                print(f"    PyMuPDF with redaction failed: {str(e)[:200]}")
            finally:
                if doc: doc.close()

        # --- Method 5: Pdfplumber (Fallback) ---
        if pdfplumber and not all_method_results:
            print(f"  Page {page_num_1_idx_str}: Trying Pdfplumber (as fallback)...")
            try:
                with pdfplumber.open(pdf_path) as pdf_doc_plumber:
                    if 0 <= page_num_0_idx < len(pdf_doc_plumber.pages):
                        page_obj = pdf_doc_plumber.pages[page_num_0_idx]
                        # You can add pdfplumber's table_settings here if needed, e.g. page_obj.extract_tables(table_settings={...})
                        plumber_raw_tables_list = page_obj.extract_tables() 
                        if plumber_raw_tables_list:
                            print(f"    Pdfplumber found {len(plumber_raw_tables_list)} potential table structure(s).")
                            for i, raw_table_data in enumerate(plumber_raw_tables_list):
                                cdf = pd.DataFrame() 
                                if raw_table_data:
                                    df_temp = pd.DataFrame(raw_table_data)
                                    if not df_temp.empty:
                                        # Promote first row to header if it seems valid
                                        if df_temp.iloc[0].notna().any():
                                            h = make_unique_headers(df_temp.iloc[0].tolist())
                                            dc = df_temp.iloc[1:].copy()
                                            try: 
                                                if not dc.empty or h : 
                                                    dc.columns = h
                                                    cdf = dc
                                                else: # only headers, no data rows or empty dc
                                                    cdf = pd.DataFrame(columns=h) if h else df_temp # fallback to original if header logic fails
                                            except Exception: cdf = df_temp # Fallback if setting columns fails
                                        else: cdf = df_temp # First row is empty or all NaNs
                                    else: cdf = df_temp # Empty raw table data
                                
                                if is_reportable_table(cdf, method_name="PDFPlumber", table_index=i):
                                    confidence = 75.0 # Assign a default confidence for Pdfplumber
                                    all_method_results.append((cdf, "Pdfplumber", confidence))
                                    print(f"      Added Pdfplumber Table {i} (Conf:{confidence:.1f})")
                    else:
                        print(f"    Pdfplumber: Page index {page_num_0_idx} out of bounds.")
            except Exception as e:
                print(f"    Pdfplumber failed: {str(e)[:200]}")

        # --- Method 6: OCR Fallback ---
        if pytesseract and fitz and OCR_METHOD_ENABLED and not all_method_results:
            print(f"  Page {page_num_1_idx_str}: Trying OCR method (as fallback)...")
            doc_ocr = None
            try:
                doc_ocr = fitz.open(pdf_path)
                if 0 <= page_num_0_idx < len(doc_ocr):
                    page_ocr = doc_ocr.load_page(page_num_0_idx)
                    pix = page_ocr.get_pixmap(dpi=300) 
                    img_bytes = pix.tobytes("png") 
                    ocr_config = r'--psm 6' 
                    print(f"    OCR: Running pytesseract on page image (config: '{ocr_config}')...")
                    ocr_text = pytesseract.image_to_string(io.BytesIO(img_bytes), lang='eng', config=ocr_config)
                    if ocr_text and not ocr_text.isspace():
                        print(f"    OCR: Text extracted (length: {len(ocr_text)}). Attempting to parse into DataFrame.")
                        df_ocr = ocr_text_to_dataframe(ocr_text)
                        if not df_ocr.empty: # ocr_text_to_dataframe might return empty
                            if is_reportable_table(df_ocr, method_name="OCR", table_index=0): # Assuming one main table from OCR per page
                                confidence = 60.0 # Assign a default, generally lower confidence for OCR
                                all_method_results.append((df_ocr, "OCR", confidence))
                                print(f"      Added OCR Table (Conf:{confidence:.1f})")
                        else: print(f"    OCR: No tables formed from OCR text.")
                    else: print(f"    OCR: No text extracted from page image or text was only whitespace.")
            except pytesseract.TesseractNotFoundError:
                print("    OCR Error: Tesseract is not installed or not found in your PATH.")
            except Exception as e:
                print(f"    OCR method failed: {str(e)[:200]}")
            finally:
                if doc_ocr: doc_ocr.close()

        # ===== SELECT BEST RESULT PER TABLE REGION =====
        final_selected_tables_for_page = [] # This will store (df, method_name, confidence) tuples
        
        if all_method_results:
            print(f"  Page {page_num_1_idx_str}: Found {len(all_method_results)} candidate table(s) from various methods. Selecting best per region...")
            table_groups = {}
            for df_candidate, method_candidate, confidence_candidate in all_method_results:
                if df_candidate is None or df_candidate.empty: continue

                # Create a hash based on table dimensions and first few cell contents of the first row
                # This helps group multiple detections of the *same* table by different methods.
                # Using more cells for hash key might improve uniqueness for tables with similar dimensions/headers.
                first_row_str = "".join(str(c)[:20] for c in df_candidate.iloc[0].fillna('').values[:min(3, df_candidate.shape[1])])
                table_hash = f"{df_candidate.shape[0]}x{df_candidate.shape[1]}:{first_row_str}"
                
                if table_hash not in table_groups:
                    table_groups[table_hash] = []
                table_groups[table_hash].append((df_candidate, method_candidate, confidence_candidate))
            
            print(f"    Grouped into {len(table_groups)} potential unique table regions.")
            table_idx_on_page = 0
            for group_hash, tables_in_group in table_groups.items():
                if tables_in_group:
                    # Select table with highest confidence from this group
                    best_table_in_group = max(tables_in_group, key=lambda x: x[2])
                    final_selected_tables_for_page.append(best_table_in_group)
                    table_idx_on_page += 1
                    print(f"    Region {table_idx_on_page}: Selected '{best_table_in_group[1]}' (Confidence: {best_table_in_group[2]:.2f}) over {len(tables_in_group)-1} other candidate(s) for this region.")
        
        # The fallback you had: if not selected_tables: selected_tables = all_method_results
        # This fallback is problematic if grouping is the primary way to de-duplicate.
        # If grouping works, final_selected_tables_for_page will have de-duplicated tables.
        # If all_method_results was empty, final_selected_tables_for_page will be empty.
        # If all_method_results had items but grouping failed to select (e.g. bug), then it would be empty.
        # For now, the logic above should populate final_selected_tables_for_page if there are any good tables.

        if final_selected_tables_for_page:
            # The structure of extracted_data_final needs to be consistent.
            # Original: (page_num_0idx, list_of_dfs, method_used_for_page)
            # New from your snippet: (page_num_0_idx, tables_found_on_page) 
            # where tables_found_on_page is list of (df, method, confidence)
            # This matches the new structure.
            extracted_data_final.append((page_num_0_idx, final_selected_tables_for_page))
            print(f"  SUMMARY for Page {page_num_1_idx_str}: Final selection of {len(final_selected_tables_for_page)} table(s).")
        else:
            print(f"  SUMMARY for Page {page_num_1_idx_str}: No tables extracted or selected for this page.")
    
    print("----------------------------------------------------------------------")
    if not extracted_data_final:
        print(f"Table Extractor: No tables were successfully extracted and selected by any method from any of the specified pages of '{os.path.basename(pdf_path)}'.")
    return extracted_data_final


# --- Main Execution Block for Table Extractor ---
if __name__ == "__main__":
    output_base_folder_name = "extracted_tables_output"
    # Check for primary parsing libraries
    if not (camelot or tabula or pdfplumber or fitz): 
        print("CRITICAL Error: At least one primary PDF parsing library (PyMuPDF, Camelot, Tabula, PDFPlumber) must be available. Exiting.")
        exit()
    if not pytesseract and OCR_METHOD_ENABLED:
        print("INFO: pytesseract OCR library is not available or Tesseract OCR engine is not installed. OCR method will be skipped if enabled as a fallback.")

    # 1. Discover PDF folder (Same as your original script)
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd() 
    current_working_dir = os.getcwd()
    identified_pdf_folder_path_ext = None
    paths_to_check_ext = [script_dir, current_working_dir, os.path.join(script_dir, "NEW"),
                          os.path.join(current_working_dir, "NEW"), os.path.join(script_dir, "PDF"),
                          os.path.join(current_working_dir, "PDF")]
    print("Table Extractor: Looking for PDF folder...")
    for p_path in paths_to_check_ext:
        abs_p_path = os.path.abspath(p_path)
        if os.path.isdir(abs_p_path):
            try:
                if any(f.lower().endswith(".pdf") for f in os.listdir(abs_p_path)):
                    identified_pdf_folder_path_ext = abs_p_path
                    print(f"Table Extractor: Successfully identified PDF folder: {identified_pdf_folder_path_ext}")
                    break
            except OSError: pass 

    if identified_pdf_folder_path_ext is None: 
        identified_pdf_folder_path_ext = os.getcwd()
        print(f"Table Extractor: Could not find specific PDF subfolder. Using current working directory as PDF folder: {identified_pdf_folder_path_ext}")
        try:
            if not any(f.lower().endswith(".pdf") for f in os.listdir(identified_pdf_folder_path_ext)):
                print(f"\nError: No PDF files found in the script's directory, 'PDF'/'NEW' subfolders, or current working directory ({identified_pdf_folder_path_ext}). Please place PDF files in one of these locations.")
                exit()
        except OSError as e:
            print(f"\nError: Could not access current working directory to check for PDFs: {e}")
            exit()

    pdf_files_list_ext = [f for f in os.listdir(identified_pdf_folder_path_ext) if f.lower().endswith(".pdf")]
    if not pdf_files_list_ext:
        print(f"No PDF files found in: {identified_pdf_folder_path_ext}"); exit()

    # 2. List PDFs and get user choice (Same as your original script)
    print("\nTable Extractor: Available PDF files:")
    for i, file_name in enumerate(pdf_files_list_ext): print(f"  {i + 1}: {file_name}")
    selected_pdf_filename_for_extractor = None
    selected_pdf_full_path_for_extractor = None
    while True:
        try:
            choice_str = input(f"Table Extractor: Enter PDF number to process (1-{len(pdf_files_list_ext)}): ")
            if not choice_str.strip(): continue
            choice = int(choice_str)
            if 1 <= choice <= len(pdf_files_list_ext):
                selected_pdf_filename_for_extractor = pdf_files_list_ext[choice - 1]
                selected_pdf_full_path_for_extractor = os.path.join(identified_pdf_folder_path_ext, selected_pdf_filename_for_extractor)
                print(f"\nTable Extractor: Selected: {selected_pdf_filename_for_extractor}")
                break
            else: print(f"Invalid choice. Enter 1-{len(pdf_files_list_ext)}.")
        except ValueError: print("Invalid input. Enter a number.")
        except Exception as e: print(f"Error during selection: {e}"); exit()

    # 3. Load target pages from the JSON file (Same as your original script)
    target_pages_0_indexed = []
    json_input_filename = os.path.splitext(selected_pdf_filename_for_extractor)[0] + "_target_pages.json"
    json_input_path = os.path.join(os.path.dirname(selected_pdf_full_path_for_extractor), json_input_filename)

    print(f"Table Extractor: Attempting to load target pages from: {json_input_path}")
    if os.path.exists(json_input_path):
        try:
            with open(json_input_path, 'r') as f:
                loaded_pages = json.load(f)
            if isinstance(loaded_pages, list) and all(isinstance(p, int) for p in loaded_pages):
                target_pages_0_indexed = loaded_pages
                print(f"Table Extractor: Successfully loaded {len(target_pages_0_indexed)} target page(s): {[p + 1 for p in target_pages_0_indexed]} (1-indexed)") 
            else:
                print(f"Table Extractor: Warning - Data in {json_input_filename} is not a valid list of integers. Will prompt for manual input.")
        except Exception as e:
            print(f"Table Extractor: Error loading target pages from {json_input_path}: {e}. Will prompt for manual input.")
    else:
        print(f"Table Extractor: Warning - Target pages file '{json_input_filename}' not found.")

    if not target_pages_0_indexed and selected_pdf_full_path_for_extractor:
        print(f"\nTable Extractor: Could not automatically load target pages for '{selected_pdf_filename_for_extractor}'.")
        while True:
            pages_input_str = input("Enter 0-indexed page numbers manually (e.g., 0,3,10 for pages 1,4,11), 'all' for all pages, or leave blank to skip: ")
            if not pages_input_str.strip():
                print("No target pages provided manually. Extraction will be skipped.")
                target_pages_0_indexed = []
                break
            if pages_input_str.lower() == 'all':
                if fitz: 
                    try:
                        temp_doc = fitz.open(selected_pdf_full_path_for_extractor)
                        num_doc_pages = len(temp_doc)
                        temp_doc.close()
                        target_pages_0_indexed = list(range(num_doc_pages))
                        print(f"Processing all {num_doc_pages} pages (0-indexed: 0 to {num_doc_pages-1}).")
                        break
                    except Exception as e:
                        print(f"Error determining page count with PyMuPDF: {e}. Please specify pages numerically.")
                else: 
                    print("PyMuPDF (fitz) library is not available to determine total page count for 'all'. Please install PyMuPDF or specify page numbers numerically.")
                    print("Please enter page numbers numerically (e.g., 0,1,2) or ensure PyMuPDF is installed.")
                    continue 
            else: 
                try:
                    target_pages_0_indexed = [int(p.strip()) for p in pages_input_str.split(',')]
                    if any(p < 0 for p in target_pages_0_indexed):
                        print("Invalid page number (negative). Page numbers must be 0 or positive.")
                        target_pages_0_indexed = [] 
                        continue 
                    print(f"Manually entered pages (0-indexed): {target_pages_0_indexed}")
                    break 
                except ValueError:
                    print("Invalid page number format. Use comma-separated numbers (e.g., 0,3,10).")
                    target_pages_0_indexed = [] 

    # Call the updated extraction function
    if selected_pdf_full_path_for_extractor and target_pages_0_indexed:
        extracted_tables_data = extract_tables_advanced(selected_pdf_full_path_for_extractor, target_pages_0_indexed)

        if extracted_tables_data:
            pdf_basename = os.path.splitext(selected_pdf_filename_for_extractor)[0]
            base_output_path = os.path.join(os.getcwd(), output_base_folder_name)
            output_dir_for_pdf = os.path.join(base_output_path, pdf_basename)

            if not os.path.exists(output_dir_for_pdf):
                os.makedirs(output_dir_for_pdf)
                print(f"Table Extractor: Created output directory: {output_dir_for_pdf}")

            num_tables_saved = 0
            print(f"\n--- Table Extractor: Saving tables from {selected_pdf_filename_for_extractor} ---")
            
            # Updated saving logic to handle the new structure of extracted_tables_data
            # extracted_tables_data is now a list of: (page_num_0idx, list_of_selected_tables)
            # where list_of_selected_tables contains (df_table, method_name, confidence_score)
            for page_num_0idx, selected_tables_on_page in extracted_tables_data:
                for table_idx_on_page, (df_table, method_name, confidence_score) in enumerate(selected_tables_on_page):
                    # Use a more descriptive name, including method and confidence
                    method_short = method_name.split("-")[0].lower().replace(" ", "") # e.g., camelotlattice, tabula, pymupdf
                    
                    # Sanitize method_short for filename (e.g. replace characters not allowed in filenames)
                    method_short_sanitized = "".join(c if c.isalnum() else "_" for c in method_short)

                    csv_name = f'page_{page_num_0idx + 1:04d}_table_{table_idx_on_page + 1:02d}_meth_{method_short_sanitized}_conf_{confidence_score:.0f}.csv'
                    csv_full_path = os.path.join(output_dir_for_pdf, csv_name)
                    try:
                        df_table.to_csv(csv_full_path, index=False, encoding='utf-8-sig')
                        num_tables_saved += 1
                    except Exception as e:
                        print(f"  Error saving to {csv_full_path}: {e}")
            
            if num_tables_saved > 0:
                print(f"\nTable Extractor: Saved {num_tables_saved} table(s) to '{output_dir_for_pdf}'.")
            else:
                print("\nTable Extractor: No tables were ultimately saved (though some may have been processed and selected).")
    elif selected_pdf_full_path_for_extractor and not target_pages_0_indexed:
        print(f"\nTable Extractor: No target pages specified for '{selected_pdf_filename_for_extractor}'. Extraction skipped.")

    print("\n--- Table Extractor Script Finished ---")
