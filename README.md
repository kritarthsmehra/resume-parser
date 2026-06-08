# Resume Parser

LLM-backed resume parser with a FastAPI backend and Streamlit UI. Accepts PDF and DOCX resumes, extracts readable text, sends it to a configured LLM, validates the structured response with Pydantic, and persists records by document ID.

## Features

| Area | Detail |
|------|--------|
| **Extraction** | Contact, summary, work experience, education, projects, technical/soft skills, languages, certifications, awards |
| **LLM providers** | OpenAI (GPT-4o-mini default), Anthropic Claude (Haiku default), Ollama (local) |
| **API** | `POST /api/upload`, `GET /api/resumes`, `GET /api/resume/{id}`, `DELETE /api/resume/{id}` |
| **Reliability** | Exponential back-off retry on transient LLM errors, atomic JSON writes, magic-byte file validation |
| **Observability** | Structured JSON logging via loguru, auto-generated OpenAPI docs at `/docs` |
| **UI** | Streamlit — upload, extract, fetch by ID, recent-parses list, one-click JSON download |

---

## Quick Start

```bash
python -m venv .venv

.venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env    
```

Edit `.env` and set at least one LLM provider (see below).

**Start the API:**
```bash
uvicorn app.main:app --reload
```

**Start the UI** (second terminal):
```bash
streamlit run ui/streamlit_app.py
```

Open:
- UI → `http://localhost:8501`
- API docs → `http://localhost:8000/docs`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RESUME_LLM_PROVIDER` | `openai` | LLM provider: `openai`, `anthropic`, or `ollama` |
| `OPENAI_API_KEY` | — | Required when provider is `openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `ANTHROPIC_API_KEY` | — | Required when provider is `anthropic` |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `MAX_UPLOAD_MB` | `10` | Maximum resume file size in MB |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-call LLM timeout |
| `LLM_MAX_RETRIES` | `3` | Retry attempts on transient errors |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `API_BASE_URL` | `http://localhost:8000` | API base URL used by the Streamlit UI |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated CORS origins |

### Provider examples

**OpenAI**
```env
RESUME_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

**Anthropic Claude**
```env
RESUME_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

**Ollama (local)**
```env
RESUME_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

---

## API Reference

### `POST /api/upload`
Upload a resume and receive extracted structured data.

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@resume.pdf"
```

**Response `201`**
```json
{
  "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename": "resume.pdf",
  "uploaded_at": "2025-06-05T10:30:00Z",
  "extracted_data": {
    "contact": { "name": "Jane Smith", "email": "jane@example.com", ... },
    "summary": "...",
    "work_experience": [...],
    "education": [...],
    "projects": [...],
    "technical_skills": [...],
    "soft_skills": [...],
    "languages": [...],
    "certifications": [...],
    "awards": [...],
    "raw_extraction_notes": [...]
  }
}
```

### `GET /api/resumes`
Paginated list of parsed resumes (newest first).

```bash
curl "http://localhost:8000/api/resumes?limit=20&offset=0"
```

### `GET /api/resume/{document_id}`
Retrieve a previously parsed resume.

```bash
curl http://localhost:8000/api/resume/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

Returns `404` if not found.

### `DELETE /api/resume/{document_id}`
Delete a resume record and its uploaded file.

```bash
curl -X DELETE http://localhost:8000/api/resume/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

Returns `204 No Content` on success, `404` if not found.

### `GET /health`
Liveness check.

---

## Project Structure

```
app/
  main.py                 FastAPI routes and error handling
  models.py               Pydantic request/response models
  services/
    llm_service.py        OpenAI / Anthropic / Ollama extraction clients
    parser.py             PDF and DOCX text extraction
    storage.py            Upload persistence and JSON record store
  utils/
    logger.py             Loguru JSON logging setup
ui/
  streamlit_app.py        Streamlit frontend
data/
  uploads/                Original uploaded files (gitignored)
  resumes/                Parsed JSON records (gitignored)
```

---

## Design Notes

- **LLM prompt** — A detailed system prompt enforces strict extraction-only rules and a typed schema description so the model never fabricates data. Anthropic responses use assistant-turn prefilling (`"{"`) to reliably produce raw JSON without markdown fences.
- **Retry logic** — `LLMService` retries on rate limits (429) and transient server errors (5xx) with exponential back-off (max 3 attempts, capped at 30 s).
- **Magic-byte validation** — File headers are checked against the declared extension so a renamed binary cannot bypass the type check.
- **Atomic writes** — Resume JSON records are written to a `.tmp` file first, then atomically replaced, preventing partial reads on concurrent requests.
- **Lazy singleton** — `LLMService` is instantiated once per process via `functools.lru_cache`, keeping the provider's HTTP connection pool alive across requests.
- **Storage** — File-based JSON store; suitable for the assignment scope. Production would swap to PostgreSQL + S3.
