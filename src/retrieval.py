from vectorstore import load_vector_store

def retrieve_relevant_chunks(query: str, top_k: int = 5, distance_threshold: float = 1.6):
    """
    Retrieves the top-k relevant document chunks with a distance threshold (L2).
    A lower score means more relevant.
    """
    vector_store = load_vector_store()
    if not vector_store:
        return []
    
    # Using similarity_search_with_score to filter out highly irrelevant chunks
    chunks_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    
    # Deduplicate and filter by score
    seen_content = set()
    unique_chunks = []
    for chunk, score in chunks_with_scores:
        if score > distance_threshold: # Filter out chunks that are too far in vector space
            continue
            
        if chunk.page_content not in seen_content:
            unique_chunks.append(chunk)
            seen_content.add(chunk.page_content)
    
    return unique_chunks[:3] # Return at most 3 unique chunks
