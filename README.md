# cyber-rag-assignment
Assigment for AI Engineer position at Datafarm

Demo: 

##  Getting Started

**1. Prerequisites**
* Python 3.11+
* Google AI Studio API Key
* UV package manager installed

**2. Installation**
1. Install UV
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
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
# Create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

5. Run the RAG pipeline (this may take a while)
```bash
uv run python src/rag_pipeline.py
```

6. Launch the Application (application will run on localhost:7860)
```bash
uv run python src/app.py
```

##  Project Structure
```plaintext
cyber-rag-assignment/
├── src/
│   ├── app.py                          # Gradio web interface
│   ├── rag_pipeline.py                 # Document processing pipeline
│   ├── retrieval.py                    # Retrieval functions
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
