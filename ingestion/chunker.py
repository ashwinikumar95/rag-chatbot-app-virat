# ingestion/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", ". "]
    )
    return splitter.split_documents(documents)
