import streamlit as st
import os
from ingestion import load_pdfs, split_documents
from vectorstore import update_vector_store, load_vector_store, clear_vector_store
from retrieval import retrieve_relevant_chunks
from generation import generate_response
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")

# Initialize session state for file uploader key
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

st.title("📄 Enterprise Knowledge Assistant")
st.markdown("---")

# Sidebar for configuration and ingestion
with st.sidebar:
    st.header("Data Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        accept_multiple_files=True, 
        type=['pdf'],
        key=f"uploader_{st.session_state.file_uploader_key}"
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            data_dir = os.path.join(os.getcwd(), "data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            # Save uploaded files and track their paths
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(data_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            
            with st.spinner("Processing documents..."):
                # Load all files in data directory to build a complete index
                all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
                docs = load_pdfs(file_paths=all_files)
                chunks = split_documents(docs)
                
                # Overwrite vector store with the complete index
                update_vector_store(chunks, overwrite=True)
                st.success(f"Processed documents and created {len(chunks)} chunks.")
                
                # Increment key to clear the file uploader
                st.session_state.file_uploader_key += 1
                st.rerun()
        else:
            st.warning("Please upload at least one PDF.")

    st.markdown("---")
    st.header("Managed Documents")
    data_dir = os.path.join(os.getcwd(), "data")
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
        if not files:
            st.info("No documents uploaded.")
        else:
            for file in files:
                col1, col2 = st.columns([4, 1])
                col1.write(file)
                if col2.button("🗑️", key=file):
                    # Delete the file
                    file_path = os.path.join(data_dir, file)
                    os.remove(file_path)
                    
                    # Rebuild the index if files remain
                    remaining_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
                    if remaining_files:
                        with st.spinner("Updating index..."):
                            remaining_file_paths = [os.path.join(data_dir, f) for f in remaining_files]
                            docs = load_pdfs(file_paths=remaining_file_paths)
                            chunks = split_documents(docs)
                            update_vector_store(chunks, overwrite=True)
                    else:
                        # No files left, clear the vector store
                        clear_vector_store()
                    
                    st.success(f"Deleted {file}")
                    st.rerun()
    else:
        st.info("No documents uploaded.")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist in history
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(source)

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please provide a Groq API Key in the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Retrieve
                relevant_chunks = retrieve_relevant_chunks(prompt)
                
                unique_sources = []
                if not relevant_chunks:
                    response = "I don't have any documents to search from. Please upload some PDFs in the sidebar."
                    unique_sources = []
                else:
                    # 2. Generate with chat history
                    response, used_indices = generate_response(prompt, relevant_chunks, chat_history=st.session_state.messages)
                    # Escape dollar signs to prevent LaTeX rendering issues
                    response = response.replace("$", "\\$")
                    
                    # Extract only the unique sources that were actually used by the bot
                    for idx in used_indices:
                        if 0 <= idx < len(relevant_chunks):
                            chunk = relevant_chunks[idx]
                            source_name = os.path.basename(chunk.metadata.get('source', 'Unknown'))
                            page_num = chunk.metadata.get('page', 'N/A')
                            source_str = f"📄 {source_name} (Page {page_num})"
                            if source_str not in unique_sources:
                                unique_sources.append(source_str)
                
                st.markdown(response)
                
                # Show sources if available
                if unique_sources:
                    with st.expander("View Sources"):
                        for source in unique_sources:
                            st.markdown(source)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": unique_sources
        })
        st.rerun()
