# ESRS / CSRD Cross-Reference Extractor 
**Python 3.11 | macOS tested**

Automates the extraction of ESRS / CSRD cross-reference tables inside annual-report PDFs.  
• Finds the most likely pages • Crops them neatly • Extracts & standardises them to CSV

---

## Key features

* One-click pipeline — run 4 notebooks in order → get analysis-ready CSVs and .TXT
* No API keys — everything runs locally  
* Smart page scoring — detect where ESRS tables hide  
* Visual preview of cropped pages  
* Isolated env (`.venv`) keeps system Python untouched  

---


## Installation (macOS · Python 3.11.5)

```bash
git clone https://github.com/vklunna/bachelor-thesis-group32-new.git 
cd bachelor-thesis-group32

python3 -m venv .venv
source .venv/bin/activate          # zsh / bash

pip install --upgrade pip
pip install -r ./bachelor-thesis-group32-folder/1_code/requirements.txt

```
### Connect to Drive to load PDF Files
1. Manually upload your credentials.json file into the following folder:
`/bachelor-thesis-group32-folder/1_code/`

2.	Then run the following script to connect and load the file:
`python3 ./bachelor-thesis-group32-folder/1_code/LOADING_IN_MEMORY.py`

3. If the user wants to upload new PDF do it in folder 0_data in Drive folder:
`https://drive.google.com/drive/folders/1SlGQq1n1mqA91433tkVwhBtw8JON6YLH?usp=share_link`



## Quick Start

1. **Copy a PDF** into the `0_data/` folder.

2. **Load the following notebooks**  
   *(Make sure to select kernel → “Python (.venv)”)*
   - `2_output/step1-extractedpages.ipynb`
   - `2_output/stage2A.ipynb`
   - `2_output/stage_2B.ipynb`
   - `2_output/stage3_standardization.py`
   - `2_output/stage4.ipynb`


3. **Standardized Table:**  
   - `2_output/standardized_merged_by_company/<company>.csv`

4. **Extracted ESG Disclosure Content:**  
   - `2_output/extracted_text/<company>_extracted_text_only.txt`


## Notebook guide — what each stage does

1. **`step1-extractedpages.ipynb`**  
    - Loads a PDF from `0_data/` and converts every page to clean text.
    - Counts ESRS-style disclosure codes, entity-specific codes, and a keyword hit-list; detects table-like layouts.
    - Scores each page with a weighted formula and sorts by total score.
    - Crops the original PDF to that range and saves it as `0_data/cropped_pdf/<name>_c.pdf`.

2. **`stage2A.ipynb`**  
    - Opens the cropped PDF from `0_data/cropped_pdf/` and walks through every page in the specified range.  
    - Extracts all table objects, then filters rows whose first cell matches an ESRS code pattern (e.g., `E1-1`, `S2-3`, `BP-1`, etc.).  
    - **Output:** one raw CSV per company `2_output/extracted_tables/<company>.csv` plus a log that lists pages where tables were found.

3. **`stage_2B.ipynb`**  
   - Cleans & normalises the raw rows  
   - Unifies code strings, parses page refs, fixes headers  
   - **Output:** `2_output/filtered_table/<company>.csv`

4. **`stage3.ipynb`**   
   - Final column mapping & deeper normalisation  
   - Merges multi-page refs, deduplicates rows, runs consistency checks  
   - **Output:** `2_output/standardized_merged_by_company/<company>.csv`

5. **`stage4.ipynb`**   
   - Reads the standardised table and opens the original PDF  
   - Retrieves and normalises the full page text around each ESRS reference
   - Extracts paragraph-level disclosure evidence based on page numbers and keywords 
   - **Output:** `2_output/extracted_text/<company>_extracted_text_only.csv`
   

