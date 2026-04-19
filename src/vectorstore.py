import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from embeddings import get_embeddings_model

VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectorstore")

def create_vector_store(chunks: List[Document], store_path: str = VECTOR_STORE_PATH):
    """
    Creates a FAISS vector store from document chunks and saves it locally.
    """
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    
    vector_store.save_local(store_path)
    return vector_store

def load_vector_store(store_path: str = VECTOR_STORE_PATH):
    """
    Loads a FAISS vector store from local storage.
    """
    embeddings = get_embeddings_model()
    if os.path.exists(os.path.join(store_path, "index.faiss")):
        return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    return None

def clear_vector_store(store_path: str = VECTOR_STORE_PATH):
    """
    Deletes the FAISS index from local storage.
    """
    if os.path.exists(store_path):
        import shutil
        shutil.rmtree(store_path)
    return True

def update_vector_store(chunks: List[Document], store_path: str = VECTOR_STORE_PATH, overwrite: bool = False):
    """
    Updates an existing FAISS vector store or creates a new one.
    If overwrite is True, the existing store is replaced.
    """
    if overwrite:
        return create_vector_store(chunks, store_path)
    
    vector_store = load_vector_store(store_path)
    if vector_store:
        vector_store.add_documents(chunks)
        vector_store.save_local(store_path)
    else:
        vector_store = create_vector_store(chunks, store_path)
    return vector_store
