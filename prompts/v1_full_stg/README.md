# Clinical Decision Support (PoC)

This project is a Proof-of-Concept (PoC) for a clinical decision support tool. It uses a RAG (Retrieval-Augmented Generation) pipeline with large language models (LLMs) to answer clinical questions based on an uploaded medical guideline document.

The application is built with Streamlit and uses a vector database (ChromaDB) to store and retrieve context from a medical guideline PDF.

## How to Run

Follow these steps to set up and run the application locally.

### 1. Prerequisites

- Python 3.8+
- `pip` for package management

### 2. Setup

First, clone the repository to your local machine:

```bash
git clone <your-repository-url>
cd clinical-decision-support-mvp
```

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

Install the required Python packages. You will need to create a `requirements.txt` file with the necessary libraries.

```bash
# Install dependencies
pip install streamlit python-dotenv openai chromadb
```

### 3. Environment Variables

The application requires an OpenAI API key to function. You need to create a `.env` file in the root of the project directory and add your key to it.

1.  Create a file named `.env` in the project root.
2.  Add your OpenAI API key to the file like this:

    ```
    OPENAI_API_KEY="sk-..."
    ```

### 4. Data Ingestion

This repository comes with two pre-ingested medical guidelines, which are already available in the `chroma_db/` directory and can be selected from the sidebar in the application:

- The Respiratory Chapter of the STG 2024
- The Full STG 2024

If you wish to use your own medical guideline document (e.g., a PDF), you must run the `ingest.py` script to process it and add it to the vector store.

### 5. Run the Application

Once the setup is complete and the data has been ingested, you can run the Streamlit application.

```bash
streamlit run app.py
```

The application should now be open and accessible in your web browser.