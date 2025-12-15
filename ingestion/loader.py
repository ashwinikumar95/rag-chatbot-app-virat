# ingestion/loader.py
from langchain_community.document_loaders import TextLoader

def load_documents(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()
