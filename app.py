import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("CRITICAL: GOOGLE_API_KEY not found in .env file!")

# 2. Initialize FastAPI App
app = FastAPI(title="Virtual Dealership Agent API", version="1.0")

# 3. Define Request Model for the API
class ChatRequest(BaseModel):
    question: str

# Global variable to hold our AI chain
qa_chain = None

@app.on_event("startup")
async def startup_event():
    global qa_chain
    print("Initializing AI components...")
    
    # A. Initialize the same embedding model used in ingestion
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    
    # B. Load the existing Chroma database
    print("Loading ChromaDB...")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # C. Create the retriever (Fetches the top 3 most relevant chunks of text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # D. Initialize Gemini LLM (Temperature 0.1 ensures factual, non-creative answers)
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2", 
    #     temperature=0.1,
    #     google_api_key=api_key
    # )
    # D. Initialize Groq LLM (Using Meta's Llama 3.1)
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    
    # E. Define the STRICT Guardrail Prompt
    # This is critical for enterprise AI. It prevents hallucinations.
    prompt_template = """
    You are a highly professional Virtual Dealership Assistant. 
    Use ONLY the following context to answer the user's question. 
    If the answer is not contained in the context, do not guess, hallucinate, or provide outside information. 
    Instead, reply EXACTLY with: 'I am sorry, I cannot answer that based on the provided dealership documents.'
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )
    
    # F. Build the modern LangChain retrieval chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    print("✅ AI Engine Ready!")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """This is the endpoint our frontend will talk to."""
    if not qa_chain:
        raise HTTPException(status_code=500, detail="AI Engine not initialized")
        
    try:
        # Run the query through the RAG chain
        response = qa_chain.invoke({"input": request.question})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Runs the server locally on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)