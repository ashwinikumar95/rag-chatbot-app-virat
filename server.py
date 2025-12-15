# server.py - FastAPI RAG Server
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from ingestion.crawler import crawl_site
from ingestion.cleaner import clean_html
from ingestion.loader import load_documents
from ingestion.chunker import create_chunks
from ingestion.embedder import embed_and_store, get_vectorstore

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Global RAG chain (initialized on startup or after crawl)
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG chain on startup if vector store exists."""
    global rag_chain
    if Path("./chroma_db").exists():
        try:
            initialize_rag_chain()
            print("✅ RAG chain initialized from existing vector store")
        except Exception as e:
            print(f"⚠️ Could not initialize RAG chain: {e}")
    yield


app = FastAPI(
    title="RAG Chatbot API",
    description="Crawl websites and ask questions using RAG",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response Models
class CrawlRequest(BaseModel):
    baseUrl: str


class AskRequest(BaseModel):
    question: str


class CrawlResponse(BaseModel):
    message: str
    baseUrl: str
    pages_crawled: int
    chunks_created: int


class AskResponse(BaseModel):
    question: str
    answer: str


def initialize_rag_chain():
    """Initialize the RAG chain from the vector store."""
    global rag_chain
    
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. 
If you cannot answer from the context, say "I don't have information about that."

Context:
{context}

Question: {question}

Answer:
""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain


# Routes
@app.get("/")
async def health():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "RAG Chatbot API is running",
        "rag_initialized": rag_chain is not None
    }


@app.post("/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest):
    """Crawl a website and ingest content into the vector store."""
    global rag_chain
    
    # Validate URL
    try:
        from urllib.parse import urlparse
        parsed = urlparse(request.baseUrl)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid baseUrl")
    
    try:
        # Step 1: Crawl the website
        print(f"🕷️ Starting crawl of {request.baseUrl}")
        pages = crawl_site(request.baseUrl)
        print(f"📄 Crawled {len(pages)} pages")
        
        if not pages:
            raise HTTPException(status_code=400, detail="No pages could be crawled")
        
        # Step 2: Clean and combine all HTML content
        all_text = []
        for page in pages:
            cleaned = clean_html(page["html"])
            if cleaned:
                all_text.append(f"# {page['title']}\n{cleaned}")
        
        combined_text = "\n\n".join(all_text)
        
        # Step 3: Save to processed file
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        clean_file = "data/processed/clean_text.txt"
        with open(clean_file, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print(f"💾 Saved clean text to {clean_file}")
        
        # Step 4: Load, chunk, and embed
        documents = load_documents(clean_file)
        chunks = create_chunks(documents)
        print(f"🧩 Created {len(chunks)} chunks")
        
        embed_and_store(chunks)
        print("✅ Stored in vector database")
        
        # Step 5: Reinitialize RAG chain
        initialize_rag_chain()
        print("✅ RAG chain reinitialized")
        
        return CrawlResponse(
            message="Crawl and ingestion completed successfully",
            baseUrl=request.baseUrl,
            pages_crawled=len(pages),
            chunks_created=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Crawl error: {e}")
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Answer a question using the RAG chain."""
    global rag_chain
    
    if not request.question:
        raise HTTPException(status_code=400, detail="question is required")
    
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No data has been ingested yet. Please crawl a website first."
        )
    
    try:
        answer = rag_chain.invoke(request.question)
        return AskResponse(question=request.question, answer=answer)
    except Exception as e:
        print(f"❌ Ask error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get answer")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

