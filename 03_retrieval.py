
import os
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

INDEX_DIR = Path("index")

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Membaca index yang sudah dibuat pada B3
embeddings = np.load(INDEX_DIR / "embeddings.npy")

chunks = json.loads(
    (INDEX_DIR / "chunks.json").read_text(
        encoding="utf-8"
    )
)


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def retrieve(question, top_k=2):
    # Membuat embedding dari pertanyaan
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
    )

    query_vector = np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )

    # Menghitung kemiripan dengan seluruh chunks
    scores = [
        cosine_similarity(query_vector, vector)
        for vector in embeddings
    ]

    # Mengurutkan berdasarkan skor tertinggi
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "chunk_id": int(i) + 1,
            "score": float(scores[i]),
            "text": chunks[i],
        }
        for i in top_indices
    ]


question = "Berapa lama proses surat aktif kuliah?"

print("Pertanyaan:", question)

results = retrieve(question, top_k=2)

for item in results:
    print(f"\n===== CHUNK {item['chunk_id']} =====")
    print(f"Similarity: {item['score']:.4f}")
    print(item["text"])
