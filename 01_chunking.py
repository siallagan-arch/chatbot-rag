
from pathlib import Path

DOC_PATH = Path("docs/akademik_demo.txt")

def split_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


text = DOC_PATH.read_text(encoding="utf-8")
chunks = split_into_chunks(text)

print("Jumlah karakter dokumen:", len(text))
print("Jumlah chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n===== CHUNK {i} =====")
    print(chunk)
