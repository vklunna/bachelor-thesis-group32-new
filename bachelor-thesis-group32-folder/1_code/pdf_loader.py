import pandas as pd
import io
import fitz  # PyMuPDF
import time
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from shared_memory import pdf_docs

# === Setup Google Drive connection ===
def setup_drive_service(credentials_path):
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

# === List all PDFs in folder ===
def list_pdf_files(drive_service, folder_id):
    """List all PDF files in a folder, including shared/team drives."""
    files = []
    page_token = None
    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            fields="nextPageToken, files(id, name, parents)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=page_token
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if not page_token:
            break
    return files

# === Save and load a single PDF ===
def load_single_pdf(drive_service, folder_id, filename):
    project_root = Path(__file__).resolve().parents[0]  # No extra nesting
    output_path = project_root.parent / "0_data" / "pdfs" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = (
        f"'{folder_id}' in parents and mimeType='application/pdf' and name='{filename}' and trashed=false"
    )
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if not files:
        print(f"❌ File '{filename}' not found.")
        return None

    file = files[0]
    fh = io.BytesIO()
    request = drive_service.files().get_media(fileId=file['id'])
    downloader = MediaIoBaseDownload(fh, request)
    while True:
        status, done = downloader.next_chunk()
        if done:
            break
    fh.seek(0)

    with open(output_path, "wb") as out_file:
        out_file.write(fh.read())

    doc = fitz.open(output_path)
    pdf_docs[file['name']] = doc
    print(f"✅ Downloaded and loaded '{file['name']}' with {doc.page_count} pages")
    return {file['name']: doc}

# === Load PDFs listed in Excel ===
def load_pdfs_from_excel(drive_service, folder_id, excel_path, sheet_name, column_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    target_pdfs = df[column_name].dropna().astype(str).str.strip().tolist()

    all_files = list_pdf_files(drive_service, folder_id)
    matching_files = [f for f in all_files if f["name"] in target_pdfs]
    matched_names = {f["name"] for f in matching_files}
    missing = [name for name in target_pdfs if name not in matched_names]

    if missing:
        print("\n⚠️ Missing files:", *missing, sep="\n• ")

    pdf_docs.clear()
    loaded = _load_files_from_drive(drive_service, matching_files)

    for name in missing[:]:
        print(f"🔁 Retrying with single load: {name}")
        result = load_single_pdf(drive_service, folder_id, name)
        if result:
            pdf_docs.update(result)
            missing.remove(name)

    return dict(pdf_docs), missing

# === Load all PDFs from the folder ===
def load_all_pdfs(drive_service, folder_id):
    all_files = list_pdf_files(drive_service, folder_id)
    return _load_files_from_drive(drive_service, all_files)

# === Helper to load files into memory and save ===
def _load_files_from_drive(drive_service, file_list):
    pdf_docs.clear()
    start = time.time()

    project_root = Path(__file__).resolve().parents[0]
    save_dir = project_root.parent / "0_data" / "pdfs"
    save_dir.mkdir(parents=True, exist_ok=True)

    for file in file_list:
        try:
            fh = io.BytesIO()
            request = drive_service.files().get_media(fileId=file['id'])
            downloader = MediaIoBaseDownload(fh, request)
            while True:
                status, done = downloader.next_chunk()
                if done:
                    break
            fh.seek(0)

            output_path = save_dir / file['name']
            with open(output_path, "wb") as out_file:
                out_file.write(fh.read())

        # Check size
            if output_path.stat().st_size < 50 * 1024:
                print(f"⚠️ File too small: {file['name']}")
                continue

            doc = fitz.open(output_path)
            pdf_docs[file['name']] = doc
            print(f"📄 Saved and loaded {file['name']} | Pages: {doc.page_count}")
        except Exception as e:
            print(f"❌ Error loading {file['name']}: {e}")

    end = time.time()
    print(f"\n🚀 Done. Loaded {len(pdf_docs)} PDFs in {int((end - start)//60)} min {int((end - start)%60)} sec")
    return pdf_docs

def clear_memory():
    pdf_docs.clear()
    print("🧹 All loaded PDFs cleared from memory.")