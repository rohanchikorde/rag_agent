import os
os.environ["CREWAI_TELEMETRY"] = "False"

import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Corrected import

from crewai import Crew, Process, Agent, Task, LLM
from crewai.tools import BaseTool

# Load environment variables from .env file
load_dotenv()

# --- Your ingestion function for PDF processing ---
def ingest_pdf_to_vectordb(pdf_file_path: str):
    """
    Loads a PDF, chunks its content, creates embeddings, and stores them in a
    ChromaDB vector store.
    """
    try:
        # 1. Load the document using PyPDFLoader
        st.info(f"Loading PDF from: {os.path.basename(pdf_file_path)}...")
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()

        # 2. Split the document into smaller, manageable chunks
        st.info("Splitting the document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        st.success(f"Split into {len(docs)} chunks.")

        # 3. Initialize the embedding model
        st.info("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 4. Create and persist the vector store with ChromaDB
        db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        db.persist() # Best practice: explicitly call persist() to save to disk
        
        st.success("PDF has been successfully processed and stored in the vector database.")
        
        return db
        
    except Exception as e:
        st.error(f"An error occurred during PDF ingestion: {e}")
        return None

# --- Define the LLM and the retriever tool ---

# 1. Configure the LLM for OpenRouter.ai using CrewAI's OpenAI client
llm = LLM(
    model="openrouter/deepseek/deepseek-chat-v3-0324:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

# 2. Define a custom tool to retrieve data from the ChromaDB vector store
class VectorDBRetriever(BaseTool):
    name: str = "Vector DB Retriever"
    description: str = (
        "A powerful tool to search for and retrieve relevant information from "
        "the local vector database. Input should be a question or a keyword."
    )
    
    def _run(self, query: str):
        if 'vectordb' not in st.session_state:
            return "Error: Vector database not initialized. Please upload and process a PDF first."
        db = st.session_state.vectordb
        docs = db.similarity_search(query, k=4) 
        results = "\n\n".join([doc.page_content for doc in docs])
        return results

# Instantiate the custom tool
vector_db_retriever_tool = VectorDBRetriever()

# --- Define the Agent and the Task ---
researcher_agent = Agent(
    role="PDF Document Researcher",
    goal="Answer user's questions accurately by retrieving information from the provided PDF document.",
    backstory=(
        "You are an expert at analyzing and summarizing information from documents. "
        "Your primary function is to use your search tool to find and synthesize key "
        "points to provide a concise and correct answer to the user."
    ),
    verbose=True,
    allow_delegation=False,
    llm=LLM(model="openrouter/deepseek/deepseek-chat-v3-0324:free", timeout=60),
    tools=[vector_db_retriever_tool]
)

def create_analysis_task(question):
    return Task(
        description=f"Analyze the document chunks to answer the following question: '{question}'. "
                    "Use the 'Vector DB Retriever' tool to find relevant information before "
                    "providing your final answer. Your response must be based *only* on the "
                    "information you find. If the information is not in the document, state that.",
        expected_output="A concise and accurate answer to the question, based on the document's content.",
        agent=researcher_agent
    )

# --- Main Streamlit App ---
st.set_page_config(page_title="Agentic RAG PDF Analyzer")
st.title("📄 Agentic RAG PDF Analyzer")
st.markdown("Upload a PDF and ask questions to the agent crew about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if uploaded_file is not None and not st.session_state.pdf_processed:
    temp_dir = "./temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_name = f"{uuid.uuid4().hex}.pdf"
    temp_file_path = os.path.join(temp_dir, temp_file_name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    if st.button("Start PDF Analysis"):
        with st.spinner("Processing PDF and preparing the knowledge base..."):
            vectordb = ingest_pdf_to_vectordb(temp_file_path)
        
        if vectordb:
            st.session_state.vectordb = vectordb
            st.session_state.pdf_processed = True
            st.rerun() # Rerun to update the UI

if st.session_state.pdf_processed:
    st.markdown("---")
    st.success("PDF processing complete! Ask your question below:")

    query = st.text_area("Your Question:", height=100)
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Agent is thinking and retrieving information..."):
                # Create a task with the user's question
                analysis_task = create_analysis_task(query)
                
                # Create the crew with the agent and the task
                document_crew = Crew(
                    agents=[researcher_agent],
                    tasks=[analysis_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                # Kickoff the crew to get the final answer
                result = document_crew.kickoff()
                
                st.success("Analysis Complete!")
                st.write("---")
                st.subheader("Final Answer:")
                st.write(result)
        else:
            st.warning("Please enter a question.")
