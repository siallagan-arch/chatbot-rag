
import os
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

DOC_PATH = Path("docs/akademik_demo.txt")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


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

print("Jumlah chunks:", len(chunks))

vectors = []

for i, chunk in enumerate(chunks, start=1):
    print(f"Membuat embedding {i}/{len(chunks)}...")

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk,
    )

    vectors.append(result.embeddings[0].values)

embeddings = np.array(vectors, dtype=np.float32)

np.save(INDEX_DIR / "embeddings.npy", embeddings)

(INDEX_DIR / "chunks.json").write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Embedding berhasil dibuat!")
print("Ukuran embeddings:", embeddings.shape)
print("Index tersimpan di folder index/")
