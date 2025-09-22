# Local RAG Demo (Aera 4B)

This directory contains a self-contained demo that lets you:
- Crawl a website
- Chunk and embed the pages into a local FAISS index
- Chat with the content using Aera 4B via a local text-generation server

It runs entirely on your machine: no external API calls are required.

## How it works (high level)
1. You provide a starting URL. The app crawls within that domain (up to a page limit).
2. The text is chunked and embedded using `jinaai/jina-embeddings-v3`.
3. A FAISS index is built and saved as `<project>.faiss` with metadata in `<project>.json`.
4. A chat UI lets you ask questions. The app retrieves top chunks and sends them to your local LLM server (LM Studio by default) to generate an answer that only uses the provided context.

---

## Prerequisites
- Python 3.10+ installed
- LM Studio (or another local server exposing OpenAI-style Completions at `http://localhost:1234/v1/completions`).
  - Recommended model in LM Studio: Aera 4B (GGUF build, e.g. instruct q4_k_m).

## Install dependencies
From this folder:
```bash
cd local_RAG
pip install -r requirements.txt
```

## Start the local LLM server (LM Studio)
1. Open LM Studio → Developer tab (green terminal icon).
2. Load a model (recommended: Aera 4B GGUF).
3. Start the server (toggle to “Running”).
4. In the right panel → Load tab, set Context Length to `24000` and click Reload.
5. Confirm the endpoint is available at `http://localhost:1234/v1/completions`.

Tip: To sanity-check the endpoint, you can run a quick request from a REST client (optional).

## Run the app
You must run from this directory so templates resolve correctly:
```bash
cd local_RAG
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open your browser at:
```
http://localhost:8000/
```

## Using the UI
- Load Existing Project:
  - If you already have saved projects (e.g. the included sample `andemili.faiss`/`andemili.json`), select one and click “Load Selected Project”.
- Crawl & Index (Build RAG):
  - Enter a starting URL and click “Start Crawling”.
  - Watch progress in real time (SSE). When complete, click “Start Chatting”.
- Chat:
  - Ask questions about the crawled content. The model will only use the provided context and cite sources as `[n]`.

## Saved projects
- After crawling, the app saves:
  - `<project>.faiss` — FAISS index
  - `<project>.json` — chunk metadata (text + source URLs)
- Project name is derived from the domain (e.g., `www.example.com` → `example`).

## API endpoints (if you want to script)
- `GET /` — returns the HTML UI
- `POST /start-crawl` — body: `{ "url": "https://example.com" }`
- `GET /crawl-progress` — Server-Sent Events (progress updates)
- `GET /list-projects` — returns an array of available project names
- `POST /load-project/{project}` — loads an existing project
- `WS /ws/chat` — WebSocket for chat streaming

Example: start a crawl
```bash
curl -X POST 'http://localhost:8000/start-crawl' \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com"}'
```

## Configuration knobs (edit `main.py`)
- `MAX_PAGES_TO_CRAWL` — hard cap on pages
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — word-based chunking
- `MODEL_NAME` — embedding model (default `jinaai/jina-embeddings-v3`)
- `TOP_K_CHUNKS` — how many chunks to retrieve
- `MAX_CONTEXT_CHARS` — budget for context sent to the LLM
- LLM endpoint — currently hardcoded to `http://localhost:1234/v1/completions` in `to_aera()`; change it there if needed

## Troubleshooting
- UI says “system is not ready”:
  - You must first crawl a website or load an existing project.
- Chat returns errors:
  - Ensure LM Studio is running and the model is loaded; check that `http://localhost:1234/v1/completions` is reachable.
- Embedding model load fails:
  - Confirm internet access and try `pip install -U sentence-transformers`.
- FAISS issues on macOS:
  - The requirement uses `faiss-cpu`. Reinstall if needed: `pip install --force-reinstall faiss-cpu`.
- Progress stuck:
  - Look at the server logs. Some pages are skipped (non-HTML). Try a different domain.

## Notes & limits
- The crawler stays on the same domain/subdomains and skips many non-HTML file types (images, PDFs, etc.).
- This is a demo. Respect robots/terms and avoid heavy crawling on production sites.

## Developer notes
- Quick sitemap test (prints discovered URLs):
  ```bash
  python main.py test
  ```
- Default server port is `8000`.
