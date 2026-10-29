
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
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
    )

    query_vector = np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )

    scores = [
        cosine_similarity(query_vector, vector)
        for vector in embeddings
    ]

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "chunk_id": int(i) + 1,
            "score": float(scores[i]),
            "text": chunks[i],
        }
        for i in top_indices
    ]


def rag_answer(question):
    results = retrieve(question, top_k=2)

    context = "\n\n".join(
        f"[Dokumen Chunk {item['chunk_id']}]\n{item['text']}"
        for item in results
    )

    prompt = f"""
Kamu adalah asisten layanan Akademik Demo.

Jawab pertanyaan pengguna hanya berdasarkan
informasi pada KONTEKS DOKUMEN.

Jika informasi tidak tersedia di konteks,
katakan bahwa informasi tidak ditemukan
dalam dokumen Akademik Demo.

KONTEKS DOKUMEN:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        contents=prompt,
    )

    return response.text or "(Tidak ada jawaban)"


question = "Berapa biaya parkir kendaraan mahasiswa di Akademik Demo?"
print("Pertanyaan:", question)
print("\nJawaban dengan RAG:")
print(rag_answer(question))
