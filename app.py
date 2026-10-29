from flask import Flask, request, jsonify, render_template

from pathlib import Path
import os
import json

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

INDEX_DIR = Path(__file__).resolve().parent / "index"

embeddings = np.load(INDEX_DIR / "embeddings.npy")

chunks = json.loads(
    (INDEX_DIR / "chunks.json").read_text(encoding="utf-8")
)


def retrieve(question, top_k=2):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    query_vector = np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(query_vector)
    )

    scores = np.divide(
        embeddings @ query_vector,
        denominator,
        out=np.zeros(len(embeddings), dtype=np.float32),
        where=denominator != 0
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices]


def rag_answer(question):
    context = "\n\n".join(retrieve(question))

    prompt = f"""
Kamu adalah asisten layanan Akademik Demo.

Jawab hanya berdasarkan konteks dokumen berikut.
Jika informasi tidak ditemukan, katakan:
"Informasi tidak ditemukan dalam dokumen Akademik Demo."

KONTEKS:
{context}

PERTANYAAN:
{question}
"""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        contents=prompt
    )

    return response.text or "Tidak ada jawaban."


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Body harus JSON"}), 400

    question = data.get("question")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Pertanyaan wajib diisi"}), 400

    try:
        answer = rag_answer(question.strip())

        return jsonify({
            "question": question,
            "answer": answer
        })

    except Exception:
        app.logger.exception("Gagal memproses pertanyaan")
        return jsonify({
            "error": "Gagal memproses pertanyaan"
        }), 500



if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

