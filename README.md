# RAG Chatbot

A simple RAG (Retrieval Augmented Generation) chatbot that crawls websites and answers questions about the content. Built with FastAPI and LangChain.

## What it does

1. You give it a URL
2. It crawls the site and stores the content in a vector database
3. You ask questions, it answers based on what it learned

Pretty straightforward.

## Setup

```bash
# Clone and cd into the project
git clone https://github.com/ashwinikumar95/rag-chatbot-app-virat.git
cd rag-chatbot-app-virat

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirement.txt
```

You'll need an OpenAI API key. Create a `.env` file:

```
OPENAI_API_KEY=sk-your-key-here
```

## Running the server

```bash
python server.py
```

Server starts at `http://localhost:3000`. You also get free API docs at `http://localhost:3000/docs` (thanks FastAPI).

## API Usage

### Crawl a website

```bash
curl -X POST http://localhost:3000/crawl \
  -H "Content-Type: application/json" \
  -d "{\"baseUrl\": \"https://en.wikipedia.org/wiki/MS_Dhoni\"}"
```

This takes a bit since it's crawling pages, extracting text, chunking, and embedding everything. You'll see progress in the terminal.

### Ask a question

```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"When did Dhoni retire from Test cricket?\"}"
```

Returns something like:

```json
{
  "question": "When did Dhoni retire from Test cricket?",
  "answer": "MS Dhoni retired from Test cricket on 30 December 2014."
}
```

### Health check

```bash
curl http://localhost:3000
```

## Project structure

```
.
├── server.py              # FastAPI server (main entry point)
├── ingestion/
│   ├── crawler.py         # Crawls websites
│   ├── loader.py          # Loads text files
│   ├── chunker.py         # Splits text into chunks
│   └── embedder.py        # Creates embeddings, stores in ChromaDB
├── data/
│   ├── raw/               # Raw HTML (if you save it)
│   └── processed/         # Cleaned text
├── chroma_db/             # Vector database (auto-created)
└── requirement.txt
```

## How the RAG pipeline works

**Ingestion (when you call /crawl):**
```
URL → crawler → clean HTML → chunk text → embed → store in ChromaDB
```

**Query (when you call /ask):**
```
Question → find similar chunks → build context → send to GPT → return answer
```

## Config

Some things you might want to tweak in the code:

- `ingestion/crawler.py`: `MAX_DEPTH=2`, `MAX_PAGES=20` - controls how deep/wide the crawler goes
- `ingestion/chunker.py`: `chunk_size=600`, `chunk_overlap=120` - chunk sizes
- `server.py`: retriever `k=3` - how many chunks to retrieve for context

## Known limitations

- Only crawls HTML pages (no PDFs, no JS-rendered content)
- Skips login/signup/cart pages automatically
- Uses OpenAI embeddings so you need an API key (and it costs money)
- Vector DB is stored locally in `chroma_db/` folder

## Tech stack

- FastAPI (API server)
- LangChain (RAG pipeline)
- ChromaDB (vector store)
- OpenAI (embeddings + GPT-4o-mini)
- BeautifulSoup (HTML parsing)

---

Built for learning RAG systems. Feel free to fork and mess around with it.
