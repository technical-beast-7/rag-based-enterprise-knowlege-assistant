from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Returns a Hugging Face embeddings model.
    """
    return HuggingFaceEmbeddings(model_name=model_name)
