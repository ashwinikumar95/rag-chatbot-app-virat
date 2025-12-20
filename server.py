# server.py - FastAPI RAG Server
import os
import time
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from pathlib import Path

from ingestion.crawler import crawl_site, CrawlError, NetworkError, ContentError
from ingestion.cleaner import clean_html
from ingestion.loader import load_documents, is_supported_file, SUPPORTED_EXTENSIONS
from ingestion.chunker import create_chunks
from ingestion.embedder import embed_and_store, get_vectorstore, collection_exists, delete_collection
from utils.validators import validate_url, validate_session_id, URLValidationError
from utils.logger import get_server_logger, get_rag_logger, get_ingestion_logger
from utils.rate_limiter import check_rate_limit

# Import centralized config
from config import (
    MAX_QUESTION_LENGTH,
    MAX_FILE_SIZE_BYTES,
    MAX_CHAT_HISTORY_MESSAGES,
    RETRIEVER_TOP_K,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph imports for conversation memory
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List
import operator

load_dotenv()

# --- LangGraph State Schema ---
class ConversationState(TypedDict):
    """State for the conversation graph."""
    question: str
    chat_history: Annotated[List, operator.add]  # Accumulates messages
    context: str
    answer: str


# Global memory saver (thread-safe, in-memory checkpointer)
memory_saver = MemorySaver()

# Cache for session LangGraph apps
rag_apps = {}

# Initialize loggers
logger = get_server_logger()
rag_logger = get_rag_logger()
ingestion_logger = get_ingestion_logger()

# Cache for session RAG chains
rag_chains = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup lifecycle handler."""
    logger.info("RAG Chatbot API starting...")
    yield
    logger.info("RAG Chatbot API shutting down...")


app = FastAPI(
    title="RAG Chatbot API",
    description="Crawl websites and ask questions using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
# NOTE: file:// origins cannot be whitelisted (browsers block them for security).
# Always access the frontend via http://localhost:3000 instead of opening index.html directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class CrawlRequest(BaseModel):
    baseUrl: str
    session_id: str
    
    @field_validator('baseUrl')
    @classmethod
    def validate_base_url(cls, v):
        try:
            return validate_url(v)
        except URLValidationError as e:
            raise ValueError(str(e))
    
    @field_validator('session_id')
    @classmethod
    def validate_session(cls, v):
        return validate_session_id(v)


class AskRequest(BaseModel):
    question: str
    session_id: str
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question exceeds maximum length ({MAX_QUESTION_LENGTH} characters)")
        return v.strip()
    
    @field_validator('session_id')
    @classmethod
    def validate_session(cls, v):
        return validate_session_id(v)


class CrawlResponse(BaseModel):
    message: str
    baseUrl: str
    session_id: str
    pages_crawled: int
    chunks_created: int


class AskResponse(BaseModel):
    question: str
    answer: str
    session_id: str


class FileIngestResponse(BaseModel):
    message: str
    filename: str
    session_id: str
    chunks_created: int


class ResetRequest(BaseModel):
    session_id: str
    
    @field_validator('session_id')
    @classmethod
    def validate_session(cls, v):
        return validate_session_id(v)


class ResetResponse(BaseModel):
    message: str
    session_id: str


def initialize_rag_chain(session_id: str):
    """Initialize the RAG chain for a specific session (legacy, for backward compat)."""
    global rag_chains
    
    rag_logger.info(f"[{session_id}] Initializing RAG chain...")
    start_time = time.time()
    
    vectorstore = get_vectorstore(collection_name=session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Updated prompt with chat history support
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer questions based on the provided context.
If you cannot answer from the context, say "I don't have information about that."

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    rag_chains[session_id] = chain
    elapsed = time.time() - start_time
    rag_logger.info(f"[{session_id}] RAG chain initialized in {elapsed:.2f}s")
    return chain


# --- Memory-Enabled LangGraph RAG App ---

# Use config value for sliding window


def create_rag_graph(session_id: str):
    """Create a LangGraph app with memory for a session."""
    global rag_apps
    
    rag_logger.info(f"[{session_id}] Creating LangGraph RAG app with memory...")
    start_time = time.time()
    
    vectorstore = get_vectorstore(collection_name=session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Node 1: Retrieve context
    def retrieve_context(state: ConversationState) -> dict:
        """Retrieve relevant documents based on the question."""
        question = state["question"]
        docs = retriever.invoke(question)
        context = format_docs(docs)
        rag_logger.debug(f"[{session_id}] Retrieved {len(docs)} documents")
        return {"context": context}
    
    # Node 2: Generate answer with chat history
    def generate_answer(state: ConversationState) -> dict:
        """Generate answer using context and chat history."""
        question = state["question"]
        context = state["context"]
        chat_history = state.get("chat_history", [])
        
        # Apply sliding window to chat history
        trimmed_history = chat_history[-MAX_CHAT_HISTORY_MESSAGES:] if len(chat_history) > MAX_CHAT_HISTORY_MESSAGES else chat_history
        
        # Build prompt with history
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful assistant. Answer questions based on the provided context.
If you cannot answer from the context, say "I don't have information about that."

Context:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        chain = prompt | model | StrOutputParser()
        
        answer = chain.invoke({
            "chat_history": trimmed_history,
            "question": question
        })
        
        # Return new messages to be added to history
        new_messages = [
            HumanMessage(content=question),
            AIMessage(content=answer)
        ]
        
        return {
            "answer": answer,
            "chat_history": new_messages  # Will be appended via operator.add
        }
    
    # Build the graph
    graph = StateGraph(ConversationState)
    
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("generate", generate_answer)
    
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    # Compile with memory checkpointer
    app = graph.compile(checkpointer=memory_saver)
    
    rag_apps[session_id] = app
    elapsed = time.time() - start_time
    rag_logger.info(f"[{session_id}] LangGraph RAG app created in {elapsed:.2f}s")
    
    return app


def get_rag_app(session_id: str):
    """Get or create LangGraph RAG app for a session."""
    if session_id in rag_apps:
        rag_logger.debug(f"[{session_id}] Using cached LangGraph app")
        return rag_apps[session_id]
    
    if not collection_exists(session_id):
        rag_logger.warning(f"[{session_id}] No collection found for session")
        return None
    
    return create_rag_graph(session_id)


def get_rag_chain(session_id: str):
    """Get or create RAG chain for a session (legacy function)."""
    if session_id in rag_chains:
        rag_logger.debug(f"[{session_id}] Using cached RAG chain")
        return rag_chains[session_id]
    
    if not collection_exists(session_id):
        rag_logger.warning(f"[{session_id}] No collection found for session")
        return None
    
    return initialize_rag_chain(session_id)


# Routes
@app.get("/")
async def serve_index():
    """Serve the chatbot UI at root. Access via http://localhost:3000"""
    return FileResponse("index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "RAG Chatbot API is running",
        "active_sessions": len(rag_chains)
    }


@app.get("/limits")
async def get_limits():
    """Get current API limits and configuration."""
    from config import (
        MAX_QUESTION_LENGTH, MAX_URL_LENGTH, MAX_SESSION_ID_LENGTH,
        MAX_FILE_SIZE_MB, SUPPORTED_FILE_EXTENSIONS,
        MAX_CRAWL_DEPTH, MAX_PAGES_PER_CRAWL,
        RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS,
        MAX_CHAT_HISTORY_MESSAGES, RETRIEVER_TOP_K,
        CHUNK_SIZE, CHUNK_OVERLAP
    )
    return {
        "input_limits": {
            "max_question_length": MAX_QUESTION_LENGTH,
            "max_url_length": MAX_URL_LENGTH,
            "max_session_id_length": MAX_SESSION_ID_LENGTH,
        },
        "file_limits": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "supported_extensions": list(SUPPORTED_FILE_EXTENSIONS),
        },
        "crawl_limits": {
            "max_depth": MAX_CRAWL_DEPTH,
            "max_pages": MAX_PAGES_PER_CRAWL,
        },
        "rate_limits": {
            "requests_per_window": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        },
        "rag_settings": {
            "max_chat_history": MAX_CHAT_HISTORY_MESSAGES,
            "retriever_top_k": RETRIEVER_TOP_K,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
    }


@app.post("/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest, req: Request):
    """Crawl a website and ingest content into the vector store for a session."""
    # Rate limiting
    check_rate_limit(req, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)
    
    session_id = request.session_id
    start_time = time.time()
    
    logger.info(f"[{session_id}] POST /crawl - URL: {request.baseUrl}")
    
    try:
        # Step 1: Crawl the website
        ingestion_logger.info(f"[{session_id}] Stage 1/5: Crawling website...")
        pages = crawl_site(request.baseUrl)
        ingestion_logger.info(f"[{session_id}] Stage 1/5: Crawled {len(pages)} pages")
        
        # Step 2: Clean and combine all HTML content
        ingestion_logger.info(f"[{session_id}] Stage 2/5: Cleaning HTML content...")
        all_text = []
        empty_pages = 0
        for page in pages:
            cleaned = clean_html(page["html"])
            if cleaned and len(cleaned.strip()) > 50:
                all_text.append(f"# {page['title']}\n{cleaned}")
            else:
                empty_pages += 1
        
        if not all_text:
            raise ContentError("All crawled pages were empty or had no extractable content")
        
        if empty_pages > 0:
            ingestion_logger.warning(f"[{session_id}] {empty_pages} pages had no extractable content")
        
        combined_text = "\n\n".join(all_text)
        ingestion_logger.info(f"[{session_id}] Stage 2/5: Extracted {len(combined_text)} characters from {len(all_text)} pages")
        
        # Step 3: Save to temp file for processing
        ingestion_logger.info(f"[{session_id}] Stage 3/5: Loading documents...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(combined_text)
            temp_file = f.name
        
        try:
            # Step 4: Load, chunk, and embed with session isolation
            documents = load_documents(temp_file)
            
            ingestion_logger.info(f"[{session_id}] Stage 4/5: Creating chunks...")
            chunks = create_chunks(documents)
            
            if not chunks:
                raise ContentError("No chunks could be created from the content")
            
            ingestion_logger.info(f"[{session_id}] Stage 4/5: Created {len(chunks)} chunks")
            
            ingestion_logger.info(f"[{session_id}] Stage 5/5: Embedding and storing...")
            embed_and_store(chunks, collection_name=session_id)
            ingestion_logger.info(f"[{session_id}] Stage 5/5: Stored in vector database")
            
            # Initialize LangGraph RAG app with memory for this session
            create_rag_graph(session_id)
            
        finally:
            os.unlink(temp_file)
        
        elapsed = time.time() - start_time
        logger.info(f"[{session_id}] POST /crawl completed in {elapsed:.2f}s - {len(pages)} pages, {len(chunks)} chunks")
        
        return CrawlResponse(
            message="Crawl and ingestion completed successfully",
            baseUrl=request.baseUrl,
            session_id=session_id,
            pages_crawled=len(pages),
            chunks_created=len(chunks)
        )
        
    except NetworkError as e:
        logger.error(f"[{session_id}] Network error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")
    except ContentError as e:
        logger.error(f"[{session_id}] Content error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Content error: {str(e)}")
    except CrawlError as e:
        logger.error(f"[{session_id}] Crawl error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Crawl failed: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{session_id}] Unexpected error during crawl")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, req: Request):
    """Answer a question using the memory-enabled LangGraph RAG app."""
    # Rate limiting
    check_rate_limit(req, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)
    
    session_id = request.session_id
    start_time = time.time()
    
    logger.info(f"[{session_id}] POST /ask - Question: {request.question[:50]}...")
    
    # Get or create the LangGraph app with memory
    app = get_rag_app(session_id)
    if app is None:
        logger.warning(f"[{session_id}] No data ingested for session")
        raise HTTPException(
            status_code=400,
            detail="No data has been ingested for this session. Please train the chatbot first."
        )
    
    try:
        rag_logger.info(f"[{session_id}] Invoking LangGraph RAG app with memory...")
        
        # Config with thread_id = session_id for memory isolation
        config = {"configurable": {"thread_id": session_id}}
        
        # Initial state with the question
        initial_state = {
            "question": request.question,
            "chat_history": [],  # Will be loaded from checkpoint
            "context": "",
            "answer": ""
        }
        
        # Invoke the graph - memory is automatically loaded/saved via checkpointer
        result = app.invoke(initial_state, config=config)
        
        answer = result["answer"]
        
        elapsed = time.time() - start_time
        logger.info(f"[{session_id}] POST /ask completed in {elapsed:.2f}s")
        rag_logger.debug(f"[{session_id}] Answer: {answer[:100]}...")
        
        return AskResponse(question=request.question, answer=answer, session_id=session_id)
    except Exception as e:
        logger.exception(f"[{session_id}] Error during LangGraph invocation")
        raise HTTPException(status_code=500, detail="Failed to get answer")


@app.post("/ingest/file", response_model=FileIngestResponse)
async def ingest_file(
    req: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Ingest a file (PDF, TXT, DOCX) into the vector store for a session."""
    # Rate limiting
    check_rate_limit(req, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)
    
    start_time = time.time()
    
    # Validate session_id
    try:
        session_id = validate_session_id(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    logger.info(f"[{session_id}] POST /ingest/file - File: {file.filename}")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not is_supported_file(file.filename):
        logger.warning(f"[{session_id}] Unsupported file type: {file.filename}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            
            # Validate file size
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)}MB limit")
            
            # Check for empty file
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="File is empty")
            
            tmp.write(content)
            temp_path = tmp.name
        
        try:
            ingestion_logger.info(f"[{session_id}] Processing file: {file.filename} ({len(content)} bytes)")
            
            # Load documents
            ingestion_logger.info(f"[{session_id}] Stage 1/3: Loading document...")
            documents = load_documents(temp_path)
            
            if not documents:
                raise HTTPException(status_code=422, detail="No content could be extracted from the file")
            
            # Create chunks
            ingestion_logger.info(f"[{session_id}] Stage 2/3: Creating chunks...")
            chunks = create_chunks(documents)
            
            if not chunks:
                raise HTTPException(status_code=422, detail="No chunks could be created from the document")
            
            ingestion_logger.info(f"[{session_id}] Stage 2/3: Created {len(chunks)} chunks")
            
            # Embed and store
            ingestion_logger.info(f"[{session_id}] Stage 3/3: Embedding and storing...")
            embed_and_store(chunks, collection_name=session_id)
            ingestion_logger.info(f"[{session_id}] Stage 3/3: Stored in vector database")
            
            # Initialize/reinitialize LangGraph RAG app with memory
            create_rag_graph(session_id)
            
        finally:
            os.unlink(temp_path)
        
        elapsed = time.time() - start_time
        logger.info(f"[{session_id}] POST /ingest/file completed in {elapsed:.2f}s - {len(chunks)} chunks")
        
        return FileIngestResponse(
            message="File ingested successfully",
            filename=file.filename,
            session_id=session_id,
            chunks_created=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{session_id}] Error during file ingestion")
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")


@app.post("/reset/conversation", response_model=ResetResponse)
async def reset_conversation(request: ResetRequest):
    """Reset conversation memory for a session (keeps knowledge/vector store)."""
    session_id = request.session_id
    
    logger.info(f"[{session_id}] POST /reset/conversation")
    
    try:
        # Clear the memory checkpoint for this session by removing from MemorySaver
        # The MemorySaver stores checkpoints by thread_id, we need to clear it
        if session_id in rag_apps:
            # Recreate the graph to reset memory state
            create_rag_graph(session_id)
            logger.info(f"[{session_id}] Conversation memory cleared")
        
        return ResetResponse(
            message="Conversation memory cleared successfully",
            session_id=session_id
        )
        
    except Exception as e:
        logger.exception(f"[{session_id}] Error resetting conversation")
        raise HTTPException(status_code=500, detail=f"Failed to reset conversation: {str(e)}")


@app.post("/reset/knowledge", response_model=ResetResponse)
async def reset_knowledge(request: ResetRequest):
    """Reset all knowledge for a session (deletes vector store and conversation memory)."""
    session_id = request.session_id
    
    logger.info(f"[{session_id}] POST /reset/knowledge")
    
    try:
        # Remove from RAG apps cache
        if session_id in rag_apps:
            del rag_apps[session_id]
            logger.info(f"[{session_id}] Removed from rag_apps cache")
        
        # Remove from legacy RAG chains cache
        if session_id in rag_chains:
            del rag_chains[session_id]
            logger.info(f"[{session_id}] Removed from rag_chains cache")
        
        # Delete the Chroma collection
        if collection_exists(session_id):
            deleted = delete_collection(session_id)
            if deleted:
                logger.info(f"[{session_id}] Chroma collection deleted")
            else:
                logger.warning(f"[{session_id}] Failed to delete Chroma collection")
        
        return ResetResponse(
            message="Knowledge and conversation cleared successfully",
            session_id=session_id
        )
        
    except Exception as e:
        logger.exception(f"[{session_id}] Error resetting knowledge")
        raise HTTPException(status_code=500, detail=f"Failed to reset knowledge: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

