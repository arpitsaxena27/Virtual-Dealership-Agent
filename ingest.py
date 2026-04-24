import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the API key from your .env file
load_dotenv()

def ingest_data():
    pdf_path = "data/THAR-ROXX-Brochure.pdf"
    
    print(f"Loading document from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 2. Chunking: Split the document into smaller pieces
    # 1000 characters per chunk, with a 200 character overlap so we don't cut sentences in half.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} individual chunks.")
    
    # 3. Embeddings: Initialize Google's embedding model
    print("Initializing Gemini Embedding Model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 4. Vector Database: Create Chroma DB and store the chunks
    print("Generating embeddings and saving to ChromaDB...")
    # This creates a local folder called "chroma_db" to save our data permanently
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("✅ Ingestion complete! Your knowledge base is ready.")

if __name__ == "__main__":
    ingest_data()