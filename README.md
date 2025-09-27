# Agentic RAG Step by Step

Hands-on repository for the RAG (Retrieval-Augmented Generation) course. It bundles a FastAPI backend, a React + Vite frontend, and background workers that index GitHub repositories so the chatbot can answer repository-specific questions.

- **Backend** – `backend/` (FastAPI, SQLAlchemy, Celery, LangGraph, Redis, Pinecone/OpenAI clients).
- **Frontend** – `frontend/` (React + Vite, Material UI, react-chatbotify).
- **Database** – SQLite for local development (`backend/sandbox.db`).
- **Message broker** – Redis (local docker/host).

---

## 1. Architecture Walkthrough

1. `POST /indexing/index` → enqueues a Celery task that clones a GitHub repo, chunks content, writes embeddings to Pinecone, and stores the `namespace` in `IndexedRepo`.
2. `GET /indexing/repos` → returns all indexed repos (`github_url`, `namespace`, `indexed_at`). The frontend stores both the URL (for display) and namespace (for chat).
3. `POST /chat/message` → FastAPI retrieves chat history, seeds the LangGraph agent with the selected namespace, and the agent retrieves documents from Pinecone before generating an answer.


---

## 2. Local Development Setup

### 2.1 Prerequisites

- Python ≥ 3.11 (3.13 recommended)
- Node.js ≥ 18
- Redis ≥ 6 (run locally or via Docker)
- (Optional) Docker, if you prefer containerized Redis/worker

### 2.2 Backend

```bash
cd backend
uv sync  # installs and locks according to pyproject/uv.lock

# Create .env if you need to override API keys (OpenAI, Pinecone, etc.)
cp .env.example .env  # update values if the file exists

# Activate the virtualenv uv created (./.venv)
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Start FastAPI (hot reload)
uvicorn app.main:app --reload

# Start Celery worker in another shell
celery -A app.core.celery_app:celery_app worker -l info
```

> **Redis**: start a local server (`redis-server`) or run `docker run -p 6379:6379 redis:7-alpine`. The broker URL is hard-coded in `app/core/celery_app.py` as `redis://localhost:6379/0`.

### 2.3 Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite serves the UI at <http://localhost:5173>. The frontend expects the backend at `VITE_API_BASE_URL` (defaults to `http://127.0.0.1:8000`).

---

## 3. Daily Workflow

1. **Index a repo** on the `/indexing` page (frontend) or with `curl`:
   ```bash
   curl -X POST http://127.0.0.1:8000/indexing/index \
        -H 'Content-Type: application/json' \
        -d '{"github_url": "https://github.com/org/repo"}'
   ```
2. **Monitor Celery** – worker logs should show `run_indexing_task`. Results are stored in Pinecone and `IndexedRepo`.
3. **Open Chat** and choose a repo. The dropdown is keyed by `namespace`, so the chat request sends `{ ..., "namespace": "org-repo-main" }`.
4. **Ask questions** – the LangGraph agent determines whether it needs retrieval, and if so, queries Pinecone using that namespace.

---

## 4. Configuration Reference

| Component | Location | Notes |
|-----------|----------|-------|
| Celery app | `backend/app/core/celery_app.py` | Broker = Redis, tasks auto-imported from `app.indexing.tasks`. |
| Celery task | `backend/app/indexing/tasks.py` | Wraps the async indexer (`asyncio.run`). On success, adds `IndexedRepo` entry. |
| DB models | `backend/app/indexing/models.py`, `backend/app/chat/models.py` | Managed with SQLAlchemy async engine. |
| API routes | `backend/app/indexing/api.py`, `backend/app/chat/api.py` | FastAPI routers. |
| Frontend API wrappers | `frontend/src/api/*.ts` | Axios clients for indexing and chat. |
| Chat UI | `frontend/src/pages/ChatBotPage.tsx` | Dropdown selects namespace; `react-chatbotify` drives the conversation. |

Environment variables (set in `.env` for backend):

```env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=...
# customize REDIS url if not localhost
``` 

---

## 5. Testing & Tooling

- **Backend**: add pytest-based tests under `backend/tests/`. (No suite is provided yet.)
- **Frontend**: Vite + React Testing Library (TBD).
- **Linters/formatters**: follow project defaults (`ruff`, `black`, `eslint`, `prettier` if you adopt them).

---

## 6. Deployment Notes

For course demos, two managed options work well:

1. **Google App Engine** (simplest): deploy backend as a standard service, run a separate worker using App Engine Flex or Cloud Workflows + Cloud Tasks for indexing jobs.
2. **Google Cloud Run + Cloud Tasks** (flexible): containerize the FastAPI API and a worker service, use Cloud Tasks instead of Celery/Redis to queue jobs, host the frontend as a static Cloud Run service or Firebase Hosting.

Remember to swap SQLite for Cloud SQL (Postgres or MySQL) and replace local Redis with Memorystore or Cloud Tasks in production.

---

## 7. Helpful Commands

```bash
# Celery control
celery -A app.core.celery_app:celery_app inspect registered
celery -A app.core.celery_app:celery_app inspect active

# FastAPI docs
open http://127.0.0.1:8000/docs

# Regenerate frontend types (if using TypeScript types from schema)
npm run lint
```

---

## 8. Troubleshooting

- **Worker silent** → ensure Redis is running and the worker is started *after* Redis (`No nodes replied` means no worker connected).
- **`greenlet_spawn` errors** → make sure all DB writes use `save_*` helpers (they commit/rollback internally) and any background job runs inside `asyncio.run`.
- **Chat says repo isn’t indexed** → verify the Celery task succeeded and `GET /indexing/repos` returns the expected namespace.

---

Happy building! Submit PRs with course improvements or additional modules.
