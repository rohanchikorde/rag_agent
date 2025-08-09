[Agentic RAG PDF Analyzer]

This project is a Streamlit web app that allows users to upload a PDF, process its contents into a vector database using ChromaDB, and then ask questions about the document. The app uses agentic retrieval-augmented generation (RAG) with CrewAI, LangChain, and HuggingFace embeddings.

## Features

- **PDF Upload & Processing:** Upload a PDF and split it into chunks for embedding.
- **Vector Database:** Store document chunks in a local ChromaDB vector store.
- **Agentic Q&A:** Ask questions about the PDF; an agent retrieves relevant information and answers using only the document’s content.
- **LLM Integration:** Uses OpenRouter.ai’s DeepSeek Chat model for language understanding.

## How It Works

1. **Upload a PDF:** The app loads and splits the PDF into manageable chunks.
2. **Embeddings & Storage:** Chunks are embedded using HuggingFace and stored in ChromaDB.
3. **Ask Questions:** Enter a question; the agent retrieves relevant chunks and answers using CrewAI.
4. **Agentic Reasoning:** The agent only uses information found in the document.

## Setup

1. **Install dependencies:**
	```bash
	pip install streamlit crewai langchain chromadb huggingface-hub python-dotenv
	```
2. **Set environment variables:**
	- Create a `.env` file with your OpenRouter API key:
	  ```
	  OPENROUTER_API_KEY=your_openrouter_api_key
	  ```
3. **Run the app:**
	```powershell
	streamlit run .\app.py
	```

## Usage

- Open the app in your browser.
- Upload a PDF file.
- Click "Start PDF Analysis" to process the document.
- Ask questions about the PDF content.

## File Structure

- `app.py` — Main Streamlit application.
- `chroma_db/` — Directory for ChromaDB vector store.
- `.env` — Environment variables (not included).

## Technologies

- **Streamlit** for UI
- **CrewAI** for agentic orchestration
- **LangChain** for document processing
- **ChromaDB** for vector storage
- **HuggingFace** for embeddings
- **OpenRouter.ai** for LLM

## License

MIT License
# rag_agent