
# Chatbot Akademik Demo Berbasis RAG

Chatbot layanan informasi akademik menggunakan Retrieval-Augmented Generation (RAG), Google Gemini API, Flask, dan Docker.

## Teknologi
- Python 3.12
- Google Gemini API
- Flask
- NumPy
- Gunicorn
- Docker dan Docker Compose

## Fitur
- Menjawab pertanyaan berdasarkan dokumen akademik.
- Menggunakan embedding dan cosine similarity untuk mencari informasi relevan.
- Memberikan jawaban ketika informasi tersedia dalam dokumen.
- Memberikan respons ketika informasi tidak ditemukan.
- Menyediakan antarmuka chatbot berbasis web.

## Menjalankan Aplikasi

1. Siapkan file `.env` yang berisi konfigurasi API Gemini.
2. Pastikan Docker Desktop berjalan.
3. Jalankan:

```bash
docker compose up --build -d
```

4. Buka aplikasi melalui browser:

http://127.0.0.1:5000

## Memeriksa Container

```bash
docker compose ps
```

## Melihat Log

```bash
docker compose logs --tail=30
```

## Menghentikan Aplikasi

```bash
docker compose down
```

## Struktur Proyek

- `app.py` — Backend Flask dan sistem RAG
- `templates/index.html` — Antarmuka chatbot
- `docs/` — Dokumen sumber akademik
- `index/` — Data chunks dan embeddings
- `Dockerfile` — Konfigurasi Docker image
- `compose.yaml` — Konfigurasi Docker Compose
- `requirements.txt` — Dependensi Python
