from pathlib import Path
from pdf_loader import setup_drive_service, load_all_pdfs

cred_path = Path("1_code/credentials.json")
folder_id = "YOUR_FOLDER_ID_HERE"

drive = setup_drive_service(str(cred_path))

print("📥 Downloading all PDFs from Drive...")
pdf_docs = load_all_pdfs(drive, folder_id)
print(f"✅ Downloaded {len(pdf_docs)} PDFs into 0_data/pdfs/")