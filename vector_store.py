import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (GROQ_API_KEY)
load_dotenv()

# Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

def get_embeddings():
    """Initialize and return the HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def process_and_store_document(file_path: str):
    """
    Loads a PDF, chunks it, and stores the embeddings in ChromaDB.
    Returns the number of chunks added.
    """
    # 1. Load document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
        # If the PDF has no extractable text (e.g., scanned image), return early
    if not chunks:
        return 0
    # 3. Store in ChromaDB
    db = Chroma.from_documents(
        chunks, 
        get_embeddings(), 
        persist_directory=CHROMA_PATH
    )
    
    return len(chunks)

def clear_database():
    """Deletes the entire Chroma database from disk."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def answer_question(question: str):
    """
    Retrieves relevant context from ChromaDB and generates an answer using Groq and LCEL.
    """
    if not os.path.exists(CHROMA_PATH):
        return "No documents have been uploaded yet. Please upload a document first."
        
    # Connect to the existing Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    
    # Create the retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    # Initialize the LLM
    llm = ChatGroq(
        temperature=0, 
        model_name=LLM_MODEL
    )
    
    # Create the prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Helper to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Create the RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Execute the chain
    return rag_chain.invoke(question)
