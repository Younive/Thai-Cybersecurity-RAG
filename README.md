# Thai-Cybersecurity-RAG

Live demo: https://thai-cybersecurity-rag.vercel.app

##  Getting Started

**1. Prerequisites**
* Python 3.11+
* OpenRouter API Key
* UV package manager installed
* unstructured required system level dependency installed
    - Tesseract (tesseract-ocr)
    - Poppler (poppler-utils)


**2. Installation**
1. Install system level dependency install

- Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- Install Tesseract, Poppler and libmagic

```bash
# for linux
apt-get install poppler-utils tesseract-ocr libmagic-dev

# for mac
brew install poppler tesseract libmagic
```


2. Clone the repository
```bash
git clone https://github.com/Younive/cyber-rag-assistant.git
cd cyber-rag-assistant
```

3. Install dependencies with UV
```bash
uv sync
```

4. Set up environment vaiables
```bash
# Create .env.local (OpenRouter)
cat > .env.local <<'EOF'
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_EMBEDDING_MODEL=baai/bge-m3
OPENROUTER_GENERAL_MODEL=google/gemini-2.5-flash
EOF
```

5. Run the RAG pipeline (this may take a while)
```bash
uv run python src/rag_pipeline.py
```

Expected output:
```bash
Initialized RAG pipeline
============================================================
RAG PIPELINE - PROCESSING ALL DOCUMENTS
============================================================

Extracting Textbook...
✓ Extracted 521 items from 46 pages

Extracting Slide Deck...
✓ Extracted 349 items from 41 slides

Extracting Thai PDF...
✓ Extracted 126 items from 77 pages

Total extracted contents: 996

Chunking all content...
Created 1055 chunks

Creating vector store...
Stored 1055 chunks in rag_knowledge_base

============================================================
PROCESSING COMPLETE
============================================================
```
If successful: You'll see a new `chroma_db/` folder created.


6. Launch the Application

A multi-turn chat frontend (`web/`) backed by a FastAPI wrapper (`src/api.py`). It reuses the retrieval + prompt + generation pipeline. Run both processes (requires `chroma_db/` already built and `OPENROUTER_API_KEY` set):

```bash
uv run uvicorn --app-dir src api:app --port 8000   # backend API
npm run dev --prefix web                            # frontend on http://localhost:3000
```

The frontend proxies `/api/*` to the backend, so the browser only talks to `:3000`.

##  Project Structure
```plaintext
cyber-rag-assignment/
├── src/
│   ├── api.py                          # FastAPI backend (serves web/ frontend)
│   ├── rag_pipeline.py                 # Document processing pipeline
│   ├── retrieval.py                    # testing and experimenting retrieval functions 
│   ├── prompt_template.py              # RAG prompt templates
│   ├── extractors/
│   │    ├── __init__.py
│   │    ├── slide_deck.py              # extract owasp-top-10.pdf
│   │    ├── textbook.py                # extract mitre-attack-philosophy-2020.pdf
│   │    └── thai_pdf.py                # extract thailand-web-security-standard-2025.pdf
|   │
│   └── vectorstore/
│       └── manage_vectorstore.py       # Vector store management
├── dataset/
│   ├── owasp-top-10.pdf               
│   ├── mitre-attack-philosophy-2020.pdf
│   └── thailand-web-security-standard-2025.pdf
├── notebook/
│   └── experiment.ipynb                # experiment.ipynb
├── chroma_db/                          # Vector database (generated)
├── pyproject.toml                      # UV project configuration
├── uv.lock                             # UV lock file
├── .env                                # Environment variables
├── architecture.pdf                    # architecture.pdf                  
└── README.md
```

## Technical Architecture
### Components

1. **Document Processing** 
    - PDF multi modal extraction using `unstrucutured[pdf]`
    - OCR support for Thai text with Tesseract
2. **Vector Store**
    - ChromaDB for effiecient similarity search
    - `baai/bge-m3` model (via OpenRouter) for multi-language Embedding
    - Collection: `rag_knowledge_base`
3. **RAG Pipeline**
    - Semantic retrieval (top-k documents)
    - Context-aware prompt building
    - Citation-enforced generation
4. **LLM Generation**
    - `google/gemini-2.5-flash` (via OpenRouter)
    - Temperature: 0.1 (factual accuracy)
    - Strict citation requirements

## Sample Result
Sample Questions and Results from this RAG system

**1. What is Broken Access Control according to OWASP?**  
#### RAG Response

Broken Access Control vulnerabilities include violation of the principle of least privilege or deny by default [Source: owasp-top-10.pdf, Page 8], where access should only be granted for particular capabilities, roles, or users, but is available to anyone [Source: owasp-top-10.pdf, Page 8]. Further vulnerabilities involve bypassing access control checks by modifying the URL (parameter tampering or force browsing), internal application state, or the HTML page, or by using an attack tool modifying API requests [Source: owasp-top-10.pdf, Page 8]. It also includes permitting viewing or editing someone else's account, by providing its unique identifier (insecure direct [Source: owasp-top-10.pdf, Page 8]. Elevation of privilege, such as acting as a user without being logged in or acting as an admin when logged in as a user, is also a vulnerability [Source: owasp-top-10.pdf, Page 8]. Metadata manipulation, like replaying or tampering with a JSON Web Token (JWT) access control token, a cookie, or a hidden field to elevate privileges or abusing JWT invalidation, is another form [Source: owasp-top-10.pdf, Page 8]. Finally, force browsing to authenticated pages as an unauthenticated user or to privileged pages is included as well [Source: owasp-top-10.pdf, Page 8].

----------------------------- 

**2. What website security controls are required by the Thailand Web Security Standard?**
#### RAG Response

This document outlines several recommendations and guidelines related to website security, but it does not explicitly list required website security controls. It references several frameworks and standards that can be used to improve website security. These include:

NIST Cybersecurity Framework (CSF) 2.0 [Source: thailand-web-security-standard-2025.pdf, Page 11]
NIST Special Publication 800-88 Guidelines for Media Sanitization [Source: thailand-web-security-standard-2025.pdf, Page 11]
OWASP (Open Worldwide Application Security Project) recommendations for website security risks [Source: thailand-web-security-standard-2025.pdf, Page 11]
National Cyber Security Committee's standard on cloud security [Source: thailand-web-security-standard-2025.pdf, Page 10]
National Cyber Security Agency's guidelines on website security for government and critical infrastructure agencies [Source: thailand-web-security-standard-2025.pdf, Page 10]
The Electronic Transactions Development Agency's recommendations on security standards for web applications [Source: thailand-web-security-standard-2025.pdf, Page 10]
Digital Government Development Agency's Government Website Standard Version 3.0 [Source: thailand-web-security-standard-2025.pdf, Page 10]
The document also mentions the importance of establishing security guidelines to protect websites and data based on the priority of assets and resources [Source: thailand-web-security-standard-2025.pdf, Page 23]. The document states that organizations should consider three fundamental security characteristics [Source: thailand-web-security-standard-2025.pdf, Page 23]. It also mentions performing vulnerability assessments of critical services, covering the security of computer systems, network infrastructure, web servers, and web applications [Source: thailand-web-security-standard-2025.pdf, Page 7].

----------------------------- 

**3. What is the difference between a Tactic and a Technique in MITRE ATT&CK?**
#### RAG Response

The basis of ATT&CK is a set of techniques and sub-techniques that represent actions adversaries perform to accomplish objectives [Source: mitre-attack-philosophy-2020.pdf, Page 16]. These objectives are represented by tactic categories [Source: mitre-attack-philosophy-2020.pdf, Page 16]. Techniques and sub-techniques are the foundation of ATT&CK and represent the individual actions adversaries take or information learned [Source: mitre-attack-philosophy-2020.pdf, Page 35]. The representation in ATT&CK balances technical detail at the technique level with context around why actions occur at the tactic level [Source: mitre-attack-philosophy-2020.pdf, Page 16].

----------------------------- 

**4. What mitigation steps does OWASP recommend for Injection vulnerabilities?**
#### RAG Response

OWASP recommends several mitigation steps for injection vulnerabilities [Source: owasp-top-10.pdf, Page 15]. The preferred option is to use a safe API, which avoids using the interpreter entirely, provides a parameterized interface, or migrates to Object Relational Mapping Tools (ORMs) [Source: owasp-top-10.pdf, Page 15]. Positive server-side input validation should be used [Source: owasp-top-10.pdf, Page 15]. For any residual dynamic queries, special characters should be escaped using the specific escape syntax for that interpreter [Source: owasp-top-10.pdf, Page 15]. Using LIMIT and other SQL controls within queries can prevent mass disclosure of records in case of SQL injection [Source: owasp-top-10.pdf, Page 15]. Source code review is the best method of detecting if applications are vulnerable to injections [Source: owasp-top-10.pdf, Page 15]; automated testing of all parameters, headers, URL, cookies, JSON, SOAP, and XML data inputs is strongly encouraged [Source: owasp-top-10.pdf, Page 15].

----------------------------- 

**5. How does MITRE describe the purpose of Persistence techniques?**
#### RAG Response

The provided documents describe persistence techniques and give examples, but do not explicitly state the purpose of persistence techniques [Source: mitre-attack-philosophy-2020.pdf, Page 16, 17]. Therefore, this information is not available in the provided documents.


## Further Optimization
- Conversation memory
- Advanced citation formatting
- better thai language content extraction
- self-host LLM
- Advanced chunking and indexing strategy

## Changelog

### 2026-07-08

**Model stack → OpenRouter**
- Migrated all models off Google GenAI onto OpenRouter (OpenAI-compatible API). Keys/models now in `.env.local`: `OPENROUTER_API_KEY`, `OPENROUTER_EMBEDDING_MODEL`, `OPENROUTER_GENERAL_MODEL`.
- Embeddings: `baai/bge-m3` (1024-dim). Chat: `google/gemini-2.5-flash`.
- Rebuilt `chroma_db/` (1035 chunks — MITRE 536 / Thailand 369 / OWASP 130).

**UI: Gradio → Next.js**
- Removed the Gradio app (`src/app.py`) and the `gradio` dependency.
- The app is now the Next.js chat frontend (`web/`) over a FastAPI wrapper (`src/api.py`). See *Launch the Application* above.

**Reliability**
- Fixed intermittent chat socket hang-ups: root cause was free-tier chat model latency (up to ~50s) tripping the Next dev proxy. Swapped to the faster `gemini-2.5-flash` (~5–7s).
- `src/api.py::/query` now catches retrieval and chat-model errors and returns proper status codes (`502` unreachable, upstream `429`/`5xx` passed through) instead of an unhandled `500`.
- Frontend `fetch` now has a 60s timeout — a slow/hung backend fails with a clear message instead of a raw `ECONNRESET`.

**Housekeeping**
- Renamed `build_gemini_rag_prompt` → `build_rag_prompt` and `GEMINI_SYSTEM_PROMPT` → `RAG_SYSTEM_PROMPT` (provider-neutral).
- `.gitignore` now excludes `.env.local`.
