"""Export paper embeddings to JSON for Swift app."""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import fitz

PAPERS_DIR = Path(__file__).parent.parent / "papers"
OUTPUT_FILE = Path(__file__).parent / "paper_embeddings.json"

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_chunks(pdf_path, chunk_size=500):
    """Extract text chunks from PDF."""
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        words = text.split()
        for i in range(0, len(words), chunk_size // 2):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 100:
                chunks.append({
                    "text": chunk[:1000],
                    "page": page_num + 1,
                    "source": pdf_path.name
                })
    doc.close()
    return chunks

def main():
    papers = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Processing {len(papers)} papers...")

    all_data = []

    for pdf_path in papers:
        print(f"  {pdf_path.name}")
        chunks = extract_chunks(pdf_path)

        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings = model.encode(texts).tolist()

            for chunk, emb in zip(chunks, embeddings):
                all_data.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "embedding": emb
                })

    print(f"\nTotal chunks: {len(all_data)}")

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_data, f)

    print(f"Saved to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
