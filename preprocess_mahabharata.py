import os
import json
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "data/Mahabharata_Unabridged.pdf")
CHUNKS_SAVE_PATH = os.getenv("CHUNKS_PATH", "data/maha_chunks.json")

# Step 1: Read and combine text from PDF
reader = PdfReader(PDF_PATH)
print(f"[INFO] Total pages in PDF: {len(reader.pages)}")

full_text = ""
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        full_text += text + "\n"
    else:
        print(f"[WARN] Page {i} had no extractable text.")

print(f"[INFO] Extracted {len(full_text)} total characters from PDF.")

# Step 2: Clean the text
def clean_text(text):
    # Remove single line breaks within paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Collapse 3+ line breaks into 2 (paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

full_text = clean_text(full_text)
print(f"[INFO] Cleaned text has {len(full_text)} characters.")

# Step 3: Split the cleaned text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_text(full_text)
print(f"[INFO] Total chunks created: {len(chunks)}")

# Step 4: Save chunks to JSON
os.makedirs(os.path.dirname(CHUNKS_SAVE_PATH), exist_ok=True)
with open(CHUNKS_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"[SUCCESS] Chunks saved to: {CHUNKS_SAVE_PATH}")
