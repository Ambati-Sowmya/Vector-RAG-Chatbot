import streamlit as st
import os
import tempfile
from vector_store import process_and_store_document, clear_database, answer_question

# Page configuration
st.set_page_config(page_title="AI Document Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 AI Document Chatbot")
st.markdown("Upload a PDF document and ask questions about it. The AI will search the document and generate an answer using **Vector Embeddings** and **Llama 3 (via Groq)**.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for file upload and management
with st.sidebar:
    st.header("Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Documents", 
        type="pdf", 
        accept_multiple_files=True,
        help="You can upload multiple PDFs. They will be stored in the Vector Database."
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Chunking and Embedding documents into ChromaDB..."):
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    # Save uploaded file to a temporary file for PyPDFLoader
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process and store the document
                        chunks_added = process_and_store_document(tmp_path)
                        if chunks_added == 0:
                            st.warning(f"Warning: No extractable text found in '{uploaded_file.name}'. It might be a scanned image.")
                        total_chunks += chunks_added
                    finally:
                        # Clean up the temporary file
                        os.unlink(tmp_path)
                
                st.success(f"Successfully added {total_chunks} text chunks to the database!")
        else:
            st.warning("Please upload at least one document first.")
            
    st.divider()
    
    if st.button("Clear Database", type="secondary"):
        with st.spinner("Deleting ChromaDB..."):
            clear_database()
            st.session_state.messages = []  # Clear chat history too
            st.success("Database cleared!")

# Main chat interface
st.subheader("Chat with your Documents")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = answer_question(prompt)
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
