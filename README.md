```
docker build -t semantic_book_recommender:1.0.0 .
docker run -p <your_port>:7860 --env-file .env -v chroma_book_data:/app/chroma_book_db semantic_book_recommender:1.0.0
```