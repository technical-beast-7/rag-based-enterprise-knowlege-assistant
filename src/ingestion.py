import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdfs(data_path: str = None, file_paths: List[str] = None) -> List[Document]:
    """
    Loads PDF documents from a directory or a specific list of file paths.
    """
    documents = []
    
    if file_paths:
        for file_path in file_paths:
            if os.path.exists(file_path):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        return documents

    if data_path and os.path.exists(data_path):
        for filename in os.listdir(data_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(data_path, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
    
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Splits documents into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""], # Improved separators
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    # Test the ingestion logic
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    docs = load_pdfs(DATA_DIR)
    print(f"Loaded {len(docs)} pages from PDFs.")
    
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
