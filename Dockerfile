# FROM python:3.12-slim AS builder

# ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
#     PIP_NO_CACHE_DIR=on

# WORKDIR /app

# # Install build deps only where needed for compiling wheels
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends build-essential \
#     && rm -rf /var/lib/apt/lists/*

# COPY pyproject.toml uv.lock ./

# # Derive a requirements file from pyproject plus runtime-only extras
# RUN python - <<'PY'
# import tomllib
# from pathlib import Path

# data = tomllib.loads(Path("pyproject.toml").read_text())
# base = data["project"]["dependencies"]
# extras = ["python-dotenv", "langchain-text-splitters"]
# existing = {d.split("==")[0].lower() for d in base}
# final = base + [e for e in extras if e.split("==")[0].lower() not in existing]
# Path("/tmp/requirements.txt").write_text("\n".join(final) + "\n")
# PY

# RUN pip install --prefix=/install --no-cache-dir -r /tmp/requirements.txt


# FROM python:3.12-slim

# ENV PYTHONUNBUFFERED=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=on \
#     PIP_NO_CACHE_DIR=on \
#     GRADIO_SERVER_NAME=0.0.0.0 \
#     GRADIO_SERVER_PORT=7860

# WORKDIR /app

# COPY --from=builder /install /usr/local

# # App source and assets
# COPY gradio_dashboard.py dashboard.css ./
# COPY src ./src
# COPY data ./data

# # Persist embeddings and CSVs outside the image if desired
# VOLUME ["/app/chroma_book_db", "/app/data"]

# EXPOSE 7860

# CMD ["python", "gradio_dashboard.py"]


FROM python:3.12-slim AS builder

# # Install build deps only where needed for compiling wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
