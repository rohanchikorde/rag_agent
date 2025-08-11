# Disable CrewAI telemetry before any other imports
import os
os.environ["CREWAI_TELEMETRY"] = "False"

import uuid
import json
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
        st.error(f"An error occurred during PDF ingestion: {str(e)}")
        return None

# --- Define the LLM and the retriever tool ---

# 1. Configure the LLM for OpenRouter.ai using CrewAI's OpenAI client
llm = LLM(
    model="openrouter/moonshotai/kimi-dev-72b:free",
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
            return json.dumps([])
        db = st.session_state.vectordb
        docs = db.similarity_search(query, k=6)
        items = []
        for i, doc in enumerate(docs, start=1):
            meta = getattr(doc, 'metadata', {}) or {}
            items.append({
                "id": f"S{i}",
                "text": doc.page_content,
                "page": meta.get("page"),
                "source": meta.get("source") or meta.get("file_path") or meta.get("path")
            })
        return json.dumps(items, ensure_ascii=False)

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
    llm=LLM(model="openrouter/moonshotai/kimi-dev-72b:free", timeout=60),
    tools=[vector_db_retriever_tool]
)

"""
Multi-agent setup: Researcher, Synthesizer, Critic
Guardrails: force grounding to retrieved content, limit steps in prompts, and require citations.
"""

# Synthesizer agent: crafts concise, user-facing answer with citations
synthesizer_agent = Agent(
    role="Answer Synthesizer",
    goal=(
        "Compose a concise, clear answer to the user's question strictly using the retrieved "
        "document content. Include inline citations like [S1], [S2] and a short 'Sources' list."
    ),
    backstory=(
        "You are excellent at summarization and formatting. You never invent facts and "
        "only use evidence provided by the Researcher."
    ),
    verbose=True,
    allow_delegation=False,
    llm=LLM(model="openrouter/moonshotai/kimi-dev-72b:free", timeout=60)
)

# Critic agent: checks grounding and clarity
critic_agent = Agent(
    role="Grounding Critic",
    goal=(
        "Evaluate the synthesized answer for factual grounding to the retrieved passages, "
        "clarity, and completeness. If ungrounded or incomplete, provide precise improvement notes."
    ),
    backstory=(
        "You are rigorous, terse, and focus on whether each claim is directly supported by the sources."
    ),
    verbose=True,
    allow_delegation=False,
    llm=LLM(model="openrouter/moonshotai/kimi-dev-72b:free", timeout=45)
)

def create_retrieval_task(question: str):
    return Task(
        description=(
            f"Retrieve the most relevant passages to answer: '{question}'. "
            "Use the Vector DB Retriever tool with multiple focused queries. "
            "Return ONLY a JSON array of objects with keys: id (S#), text, page, source. "
            "Do not add commentary. Limit to top 3-6 snippets from the user's uploaded PDF embeddings only."
        ),
        expected_output=(
            "A JSON array like [{\"id\":\"S1\",\"text\":\"...\",\"page\":1,\"source\":\"...\"}, ...]."
        ),
        agent=researcher_agent
    )

def _make_evidence_block(evidence_json_text: str) -> str:
    try:
        data = json.loads(evidence_json_text)
        if not isinstance(data, list) or len(data) == 0:
            return ""
        lines = []
        for item in data:
            sid = item.get("id") or "S?"
            page = item.get("page")
            src = item.get("source")
            txt = (item.get("text") or "").strip()
            head = f"[{sid}] (page {page})" if page is not None else f"[{sid}]"
            if src:
                head += f" — {src}"
            # Keep each snippet short in prompt to control length
            snippet = txt if len(txt) <= 800 else (txt[:800] + " …")
            lines.append(f"{head}:\n{snippet}")
        return "\n\n".join(lines)
    except Exception:
        return (evidence_json_text or "").strip()

def create_synthesis_task_from_evidence(question: str, evidence_block: str):
    return Task(
        description=(
            "You are given evidence snippets extracted from the user's uploaded PDF embeddings.\n\n"
            "[EVIDENCE]\n" + evidence_block + "\n\n"
            f"Using only the EVIDENCE above, write a concise answer to: '{question}'. "
            "Rules: 1) Do not add external facts. 2) Cite claims inline with [S#] (use the IDs in EVIDENCE). 3) "
            "If the answer is not in the EVIDENCE, explicitly say so. 4) End with a 'Sources' list mapping [S#] to a 1-line snippet. "
            "Keep answer under 200 words."
        ),
        expected_output=(
            "A grounded answer under 200 words with inline [S#] citations and a 'Sources' list."
        ),
        agent=synthesizer_agent
    )

def create_critique_task(synthesis_task: Task):
    return Task(
        description=(
            "Review the synthesized answer. Check that every claim is supported by the cited [S#]. "
            "If issues exist, list exact sentences and what evidence is missing or mis-cited. "
            "If the answer is acceptable, reply 'APPROVED'. Keep under 120 words."
        ),
        expected_output=(
            "Either 'APPROVED' or a short bullet list of corrections needed."
        ),
        agent=critic_agent,
        context=[synthesis_task]
    )

# --- Utilities to normalize CrewAI result objects ---
def _extract_result_text(result) -> str:
    try:
        # Common CrewAI result patterns
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        # Pydantic or attr-like objects
        for attr in ("final_output", "summary", "output", "result", "response"):
            if hasattr(result, attr):
                val = getattr(result, attr)
                if isinstance(val, (str, bytes)):
                    return val.decode() if isinstance(val, bytes) else val
        if isinstance(result, dict):
            for key in ("final_output", "summary", "output", "result", "response"):
                if key in result and isinstance(result[key], str):
                    return result[key]
        # Fallback to stringification
        return str(result)
    except Exception:
        return str(result)

# --- Main Streamlit App ---
st.set_page_config(page_title="Agentic RAG PDF Analyzer")

st.title("📄 Agentic RAG PDF Analyzer")
st.sidebar.markdown("Upload a PDF and ask questions to the agent crew about its content.")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if uploaded_file is not None and not st.session_state.pdf_processed:
    temp_dir = "./temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_name = f"{uuid.uuid4().hex}.pdf"
    temp_file_path = os.path.join(temp_dir, temp_file_name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")

    if st.sidebar.button("Start PDF Analysis"):
        # Clear chroma_db before processing new PDF
        chroma_dir = "./chroma_db"
        if os.path.exists(chroma_dir):
            import shutil
            import stat
            import time
            
            def make_writable(path):
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            
            def on_rm_error(func, path, exc_info):
                # Make file writable and try again
                make_writable(path)
                try:
                    # If it's a directory, make all files inside writable first
                    if os.path.isdir(path):
                        for root, dirs, files in os.walk(path):
                            for d in dirs:
                                make_writable(os.path.join(root, d))
                            for f in files:
                                make_writable(os.path.join(root, f))
                    # Try the operation again
                    func(path)
                except Exception as e:
                    # If still failing, log and continue
                    print(f"Could not remove {path}: {str(e)}")
            
            # Try multiple times with delay
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(chroma_dir, onerror=on_rm_error)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        st.error(f"Failed to clear ChromaDB directory after {max_retries} attempts. Please try again. Error: {str(e)}")
                        st.stop()
                    time.sleep(0.5)  # Wait before retry
        
        # Ensure the directory exists and is empty
        try:
            os.makedirs(chroma_dir, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to create ChromaDB directory: {str(e)}")
            st.stop()
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
            if 'vectordb' not in st.session_state:
                st.warning("Vector database not initialized. Please upload a PDF and click 'Start PDF Analysis' first.")
            else:
                # Phase 1: Retrieval only (structured JSON from embeddings)
                with st.spinner("Researcher retrieving passages from embeddings..."):
                    try:
                        retrieval_task = create_retrieval_task(query)
                        crew_retrieval = Crew(
                            agents=[researcher_agent],
                            tasks=[retrieval_task],
                            process=Process.sequential,
                            verbose=True
                        )
                        retrieval_result = crew_retrieval.kickoff()
                        evidence_json_text = _extract_result_text(retrieval_result)
                        evidence_block = _make_evidence_block(evidence_json_text)
                    except Exception as e:
                        st.error(f"Agent run failed (retrieval): {str(e)}")
                        evidence_block = ""

                # Phase 2: Synthesis using only EVIDENCE block
                answer_clean = ""
                if evidence_block:
                    with st.spinner("Synthesizer composing grounded answer..."):
                        try:
                            synthesis_task = create_synthesis_task_from_evidence(query, evidence_block)
                            crew_synthesis = Crew(
                                agents=[synthesizer_agent],
                                tasks=[synthesis_task],
                                process=Process.sequential,
                                verbose=True
                            )
                            result_synth = crew_synthesis.kickoff()
                            answer_text = _extract_result_text(result_synth)
                            answer_clean = answer_text.replace('```markdown', '').replace('```', '').strip()
                        except Exception as e:
                            st.error(f"Agent run failed (synthesis): {str(e)}")

                # Phase 3: Critic review (optional)
                verdict_clean = ""
                if answer_clean:
                    with st.spinner("Critic reviewing grounding and clarity..."):
                        try:
                            critic_prompt = (
                                "You are given the following synthesized answer to review.\n\n"
                                "[ANSWER]\n" + answer_clean + "\n\n"
                                "Review the answer. Check that every claim is supported by the cited [S#]. "
                                "If issues exist, list exact sentences and what evidence is missing or mis-cited. "
                                "If acceptable, reply 'APPROVED'. Keep under 120 words."
                            )
                            critic_task_ad_hoc = Task(
                                description=critic_prompt,
                                expected_output="'APPROVED' or a short bullet list of corrections.",
                                agent=critic_agent
                            )
                            crew_phase2 = Crew(
                                agents=[critic_agent],
                                tasks=[critic_task_ad_hoc],
                                process=Process.sequential,
                                verbose=True
                            )
                            result_phase2 = crew_phase2.kickoff()
                            verdict_text = _extract_result_text(result_phase2)
                            verdict_clean = verdict_text.replace('```markdown', '').replace('```', '').strip()
                        except Exception as e:
                            st.warning(f"Critic check failed: {str(e)}")

                # Render
                st.success("Analysis Complete!")
                st.write("---")
                st.subheader("Final Answer:")
                if answer_clean:
                    st.markdown(answer_clean)
                else:
                    if not evidence_block:
                        st.info("No evidence found in your uploaded PDF embeddings for this query. Try rephrasing or uploading a different file.")
                    else:
                        st.info("No concise answer found. The agents may need a more specific question.")

                if verdict_clean:
                    st.write("---")
                    st.subheader("Critic Verdict:")
                    st.markdown(verdict_clean)

                st.write("---")
                st.subheader("Full Agent Response:")
                if answer_clean:
                    st.markdown(answer_clean)
                else:
                    st.info("No full response available.")

                if answer_clean:
                    st.download_button(
                        label="📥 Download Full Response",
                        data=answer_clean,
                        file_name="agent_response.md",
                        mime="text/markdown"
                    )
        else:
            st.warning("Please enter a question.")
