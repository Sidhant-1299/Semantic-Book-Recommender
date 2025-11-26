# Semantic Book Recommender

A simple, interactive Gradio dashboard that recommends books based on meaning and mood. Each book’s description is embedded with OpenAI models, stored in a Chroma vector DB, and served through an easy web interface.

## Features
- Semantic search over book descriptions using OpenAI embeddings and Chroma.
- Filter by category and prioritize by emotional tone (happy, surprising, angry, suspenseful, sad).
- Modal gallery that shows richer book details on click.
- Docker workflow with a persistent volume for the vector database.

## Prerequisites
- Docker
- An `.env` file with your `OPENAI_API_KEY` (and any other secrets you rely on).
- Data files in `data/final/` (e.g., `books_with_emotions.csv`, `tagged_description.txt`) present inside the image or mounted in.

## Quick start (Docker)
Build the image from the repo root:
```
docker build -t semantic_book_recommender:1.0.0 .
```

Run the app (exposing Gradio on port 7860, loading env vars, and persisting Chroma index):
```
docker run \
  -p 7860:7860 \
  --env-file .env \
  -v chroma_book_data:/app/chroma_book_db \
  semantic_book_recommender:1.0.0
```

- Change the left side of `-p` to expose on a different host port.
- The named volume `chroma_book_data` keeps embeddings across runs; speeds up consecutive runs; remove it if you want a fresh index.
- Ensure your `.env` file is in the same directory you run the command from (or provide an absolute path).

## Local development (optional)
- Populate a `.env` with `OPENAI_API_KEY=<your-key>`.
- Install dependencies with your preferred tool (`uv pip install -r requirements.txt` works if you use uv).
- Run the dashboard:
```
python gradio_dashboard.py
```

## Project structure
- `gradio_dashboard.py` — main Gradio app.
- `dashboard.css` — custom styling for the UI.
- `chroma_book_db/` — persisted vector store (mounted as a Docker volume in production).
- `data/final/` — CSVs and text files used for embeddings and recommendations.

## License
MIT License 
