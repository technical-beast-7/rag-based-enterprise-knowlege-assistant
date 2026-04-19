import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def generate_response(query: str, context_chunks: list, chat_history: list = None):
    """
    Generates a response from the LLM using the provided context and chat history.
    Returns a tuple (response_text, used_source_indices).
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Format context with IDs for citation
    context_text = ""
    for i, doc in enumerate(context_chunks):
        context_text += f"[{i+1}] (Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}, Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}\n\n"
    
    system_prompt = f"""
    You are an Enterprise Knowledge Assistant. Use the provided context and conversation history to answer the user's question.
    If you don't know the answer based on the context, say that you don't know. Do not try to make up an answer.
    
    IMPORTANT: 
    - Keep the response concise and professional.
    - Use ONLY plain text. 
    - Do NOT use LaTeX, dollar signs ($), or any mathematical formatting. 
    - Avoid using special fonts or symbols that might disrupt the UI.
    - If you are using information from a specific context piece, cite it implicitly.
    - At the VERY end of your response, list the IDs of the context pieces you actually used to answer the question in the following format: "SOURCES: [1, 2]". 
    - If you say you don't know the answer, do not include the "SOURCES:" line.

    Retrieved Context:
    {context_text}
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history if available
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current user query
    messages.append({"role": "user", "content": query})
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )
    
    raw_response = chat_completion.choices[0].message.content
    
    # Extract SOURCES from the end of the response
    used_indices = []
    source_match = re.search(r"SOURCES:\s*\[([\d,\s]+)\]", raw_response)
    if source_match:
        try:
            used_indices = [int(idx.strip()) - 1 for idx in source_match.group(1).split(",")]
            # Remove the SOURCES line from the final response text
            clean_response = re.sub(r"\n*SOURCES:\s*\[[\d,\s]+\]", "", raw_response).strip()
        except:
            clean_response = raw_response.strip()
    else:
        clean_response = raw_response.strip()
        
    return clean_response, used_indices
