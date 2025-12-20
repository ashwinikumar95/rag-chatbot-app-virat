# ingestion/embedder.py
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "./chroma_db"


def get_embeddings():
    """Get OpenAI embeddings instance."""
    return OpenAIEmbeddings()


def embed_and_store(chunks, collection_name: str = None, persist_directory=PERSIST_DIRECTORY):
    """Embed chunks and store in ChromaDB with optional collection name for session isolation."""
    embeddings = get_embeddings()

    kwargs = {
        "documents": chunks,
        "embedding": embeddings,
        "persist_directory": persist_directory
    }
    if collection_name:
        kwargs["collection_name"] = collection_name

    vectorstore = Chroma.from_documents(**kwargs)
    return vectorstore


def get_vectorstore(collection_name: str = None, persist_directory=PERSIST_DIRECTORY):
    """Load existing vectorstore from disk with optional collection name."""
    embeddings = get_embeddings()
    
    kwargs = {
        "persist_directory": persist_directory,
        "embedding_function": embeddings
    }
    if collection_name:
        kwargs["collection_name"] = collection_name
    
    vectorstore = Chroma(**kwargs)
    return vectorstore


def collection_exists(collection_name: str, persist_directory=PERSIST_DIRECTORY) -> bool:
    """Check if a collection exists in the vectorstore."""
    try:
        vs = get_vectorstore(collection_name=collection_name, persist_directory=persist_directory)
        count = vs._collection.count()
        return count > 0
    except Exception:
        return False


def delete_collection(collection_name: str, persist_directory=PERSIST_DIRECTORY) -> bool:
    """Delete a collection from the vectorstore."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_directory)
        client.delete_collection(name=collection_name)
        return True
    except Exception:
        return False
