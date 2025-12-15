# ingestion/embedder.py
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "./chroma_db"


def get_embeddings():
    """Get OpenAI embeddings instance."""
    return OpenAIEmbeddings()


def embed_and_store(chunks, persist_directory=PERSIST_DIRECTORY):
    """Embed chunks and store in ChromaDB."""
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore


def get_vectorstore(persist_directory=PERSIST_DIRECTORY):
    """Load existing vectorstore from disk."""
    embeddings = get_embeddings()
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore
