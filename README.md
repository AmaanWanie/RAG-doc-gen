# RAG Document Generator — Proof of Concept

A standalone Retrieval-Augmented Generation (RAG) document generation application built with Streamlit, ChromaDB, sentence-transformers, and either Ollama or OpenAI.

The application allows a user to upload source documents, build a local retrieval index, define a document template at runtime, and generate a structured Markdown document section by section with source citations.

---

## 1. Project Objective

This project is a greenfield Proof of Concept for a prompt-driven RAG pipeline.

The goal is to generate structured documents from uploaded source material. The user controls the generation structure at runtime through a template editor. The application does not hardcode report sections. Instead, the user defines:

- A global base prompt
- Document sections
- Optional section-specific instructions
- Optional subsections
- Saved/reloadable templates

For each section, the system retrieves relevant source chunks from the vector database, builds a final prompt, calls an LLM, and displays the generated section with citations.

---

## 2. Key Features

- Upload one or more PDF, TXT, or Markdown files
- Extract text from uploaded documents
- Split extracted content into configurable chunks
- Generate local embeddings using sentence-transformers
- Store and query embeddings using ChromaDB
- Configure prompt templates from the Streamlit UI
- Add section-specific instructions and subsections
- Save and reload templates as JSON
- Preview final prompts before generation
- Generate one section for testing
- Generate the full document section by section
- Display source citations for every generated section
- Save generated Markdown output locally
- Download generated Markdown from the UI
- Supports local Ollama models and OpenAI chat models

---

## 3. Technology Stack

| Component | Implementation |
|---|---|
| UI | Streamlit |
| Language | Python 3.11 |
| Package Manager | uv |
| PDF Extraction | PyMuPDF |
| Chunking | Custom text splitter |
| Embeddings | sentence-transformers |
| Vector Database | ChromaDB |
| Local LLM | Ollama |
| Cloud LLM | OpenAI |
| Template Storage | JSON |
| Output Format | Markdown |

---

## 4. Architecture Overview

The system is organized as a modular RAG pipeline.

```text
Uploaded Documents
        |
        v
Document Loader
        |
        v
Text Extraction
        |
        v
Text Splitter / Chunker
        |
        v
Embedding Model
        |
        v
ChromaDB Vector Store
        |
        v
Retriever
        |
        v
Prompt Builder
        |
        v
LLM Client
        |
        v
Markdown Renderer + Citations
```

### 4.1 Document Ingestion

The ingestion layer accepts uploaded files from the Streamlit UI. Supported formats are:

- PDF
- TXT
- MD

PDF text is extracted page by page using PyMuPDF. Text files are read directly. Extracted text is stored with source metadata such as file name and page number.

Relevant modules:

```text
src/ingestion/document_loader.py
src/ingestion/text_splitter.py
src/ingestion/ingestion_pipeline.py
```

### 4.2 Chunking

The extracted text is split into searchable chunks. The user can configure:

- Chunk size
- Chunk overlap

The chunking logic is designed to avoid unnecessary word cuts and preserve readable text blocks where possible.

Each chunk stores:

- Source file
- Page number
- Chunk index
- Chunk ID
- Chunk text
- Character count

### 4.3 Embeddings

Each chunk is converted into a vector embedding using a local sentence-transformers model.

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Relevant module:

```text
src/vectorstore/embeddings.py
```

### 4.4 Vector Store

Embeddings and metadata are stored locally in ChromaDB.

Runtime ChromaDB files are stored in:

```text
data/chroma_db/
```

This folder is ignored by Git because the vector index is runtime data and can be regenerated from uploaded documents.

Relevant module:

```text
src/vectorstore/chroma_store.py
```

### 4.5 Retrieval

For each section, the retriever builds a query using the section title, section instruction, and subsection information. It retrieves the most relevant chunks from ChromaDB.

The app supports:

- Top-K retrieval
- Automatic distance filtering
- Manual distance threshold filtering

Relevant module:

```text
src/retrieval/retriever.py
```

### 4.6 Prompt Composition

The prompt builder combines:

```text
Base prompt
+ Section title
+ Section-specific instruction
+ Subsections
+ Retrieved context chunks
+ Strict generation rules
```

The base prompt is applied to every section. Section-specific instructions extend the base prompt and do not replace it.

Relevant module:

```text
src/generation/prompt_builder.py
```

### 4.7 LLM Generation

The generation layer supports multiple providers:

- `ollama`
- `openai`
- `mock`

Ollama is recommended for local testing. OpenAI can be used by adding an API key to `.env`.

Relevant modules:

```text
src/generation/llm_client.py
src/generation/generation_pipeline.py
```

### 4.8 Citation Formatting

The app adds citations programmatically after generation. The LLM is not trusted to create citations by itself.

Each generated section includes a source list showing:

- Source file
- Page number
- Chunk number

Relevant module:

```text
src/output/citation_formatter.py
```

---

## 5. Project Structure

```text
rag_document_generator/
├── app.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .streamlit/
│   └── config.toml
├── data/
│   ├── uploads/
│   ├── chroma_db/
│   ├── templates/
│   └── outputs/
├── sample_templates/
├── src/
│   ├── config.py
│   ├── models.py
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   ├── text_splitter.py
│   │   └── ingestion_pipeline.py
│   ├── vectorstore/
│   │   ├── embeddings.py
│   │   └── chroma_store.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── templates/
│   │   └── template_manager.py
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   ├── llm_client.py
│   │   └── generation_pipeline.py
│   └── output/
│       ├── citation_formatter.py
│       └── markdown_renderer.py
└── tests/
```

---

## 6. Setup Instructions

### 6.1 Prerequisites

Install the following before running the project:

- Python 3.10 or newer
- uv package manager
- Git
- Ollama, if using local LLM generation

This project was developed with Python 3.11.

to install Ollama in Windows using powershell
```powershell
winget install -e --id Ollama.Ollama
```
for MacOS/linux based:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

to install UV package manager in using powershell
```powershell
winget install -e --id astral-sh.uv
```
for MacOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check Python:

```powershell
python --version
```

Check uv:

```powershell
uv --version
```

---

### 6.2 Clone the Repository

```powershell
git clone https://github.com/AmaanWanie/RAG-doc-gen.git
cd RAG-doc-gen
```

---

### 6.3 Install Dependencies

Using uv:

```powershell
uv sync
```

If the virtual environment is not created automatically, run:

```powershell
uv venv
uv sync
```

---

### 6.4 Create Environment File

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure values as needed.

Recommended local configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=ollama
OPENAI_MODEL=gpt-4o-mini

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

CHROMA_DB_DIR=data/chroma_db
UPLOAD_DIR=data/uploads
TEMPLATE_DIR=data/templates
OUTPUT_DIR=data/outputs
```

For local-only usage, an OpenAI key is not required if `LLM_PROVIDER=ollama`.

---

### 6.5 Set Up Ollama

Install Ollama separately, then pull a local model:

```powershell
ollama pull llama3.2:3b
```

Check that Ollama is available:

```powershell
ollama list
```

The app expects Ollama to be running on:

```text
http://localhost:11434
```

If Ollama is not running, start it:

```powershell
ollama serve
```

If the command says the port is already in use, Ollama is likely already running.

---

### 6.6 Run the Streamlit App

```powershell
uv run streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## 7. How to Use the Application

### Step 1 — Upload, Extract, and Chunk Source Documents

1. Upload one or more PDF, TXT, or Markdown files.
2. Choose chunk size and chunk overlap.
3. Click:

```text
Upload, Extract and Create Chunks
```

Recommended settings:

```text
Chunk size: 1000
Chunk overlap: 200
```

After processing, the UI shows document, page, and chunk counts.

Optional review buttons:

- Review Extracted Text
- Review Chunks

These open review dialogs without cluttering the main page.

---

### Step 2 — Build Retrieval Index

Click:

```text
Build Search Index / Create Vector Index
```

The app embeds the chunks and stores them in ChromaDB.

Optional review button:

```text
Test Retrieval
```

Use this to test whether queries retrieve relevant chunks.

Example retrieval queries:

```text
system architecture
evaluation metrics
privacy risks
deployment plan
```

---

### Step 3 — Configure Prompt Template

Use the template editor to define:

- Template name
- Base prompt
- Sections
- Section-specific instructions
- Subsections

The base prompt is applied to every generated section.

Example base prompt:

```text
Write in a professional technical report style. Use only the retrieved context.
Do not invent facts. Keep the output clear, structured, and suitable for a
proof-of-concept project report.
```

Example section:

```text
Title:
System Architecture

Instruction:
Explain the main components and how they work together.

Subsection:
Core Modules

Subsection instruction:
Describe ingestion, retrieval, prompt building, generation, and output rendering.
```

The template can be saved and reloaded across sessions.

---

### Step 4 — Preview Final Prompt

The prompt preview step shows exactly what will be sent to the LLM for a selected section.

It includes:

- Base prompt
- Section instruction
- Subsection instructions
- Retrieved context chunks
- Generation rules

This step is useful for debugging and verifying that the RAG context is correct.

---

### Step 5 — Generate One Section

Before generating a full document, test one section.

Choose:

- Section
- LLM provider
- Top-K chunks
- Automatic or manual distance filtering

Recommended settings for small documents:

```text
Provider: ollama
Top-K: 2
Auto distance filter: enabled
```

Click:

```text
Generate Selected Section
```

The output should include generated content and source citations.

---

### Step 6 — Generate Full Document

After the single-section output looks correct, generate the complete document.

Click:

```text
Generate Full Document
```

The app generates each section in order, assembles the full Markdown document, saves it locally, and displays a download button.

Generated outputs are stored in:

```text
data/outputs/
```

---

## 8. Template JSON Format

Templates are stored as JSON.

Example structure:

```json
{
  "template_name": "Smart Classroom AI Report",
  "base_prompt": "Write in a professional technical report style. Use only the retrieved context. Do not invent facts.",
  "sections": [
    {
      "title": "Executive Summary",
      "instruction": "Summarize the purpose and value of the system.",
      "subsections": []
    },
    {
      "title": "System Architecture",
      "instruction": "Explain the main components and how they work together.",
      "subsections": [
        {
          "title": "Core Modules",
          "instruction": "Describe ingestion, retrieval, prompt building, generation, and output rendering."
        }
      ]
    },
    {
      "title": "Risks and Future Scope",
      "instruction": "Discuss limitations, risks, deployment considerations, and future improvements.",
      "subsections": []
    }
  ]
}
```

At least one sample template is included under:

```text
sample_templates/
```

Templates created from the UI are saved under:

```text
data/templates/
```

---

## 9. Configuration

The main configuration file is:

```text
src/config.py
```

Runtime values are read from `.env`.

Important variables:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | Default provider: `ollama`, `openai`, or `mock` |
| `OPENAI_API_KEY` | Required only for OpenAI generation |
| `OPENAI_MODEL` | OpenAI model name |
| `OLLAMA_BASE_URL` | Ollama server URL |
| `OLLAMA_MODEL` | Ollama model name |
| `CHROMA_DB_DIR` | ChromaDB persistence directory |
| `UPLOAD_DIR` | Uploaded file storage directory |
| `TEMPLATE_DIR` | Saved template directory |
| `OUTPUT_DIR` | Generated Markdown output directory |

---

## 10. Runtime Data and Git Ignore Policy

The following folders contain runtime-generated files and are not committed to GitHub:

```text
data/uploads/
data/chroma_db/
data/outputs/
```

These files can be regenerated by running the app.

The repository keeps the folder structure using `.gitkeep` files.

The real `.env` file is ignored because it may contain secrets. Only `.env.example` is committed.

## 11. Proof-of-Concept Scope

Included in this POC:

- Local Streamlit UI
- PDF/text upload
- Chunking
- Embedding
- ChromaDB retrieval
- Runtime prompt template editor
- Template save/load
- Section-by-section generation
- Source citations
- Markdown output

Not included:

- User authentication
- Multi-user workspace
- Production deployment
- Containerization
- DOCX/PDF export
- OCR for scanned PDFs
- Role-based access control

---

## 12. Example End-to-End Workflow

Recommended quick test:

1. Start Ollama and make sure `llama3.2` is available.
2. Run the Streamlit app.
3. Upload a test PDF.
4. Use chunk size `1000` and overlap `200`.
5. Click `Upload, Extract and Create Chunks`.
6. Review extracted text if needed.
7. Build the retrieval index.
8. Test retrieval with `system architecture`.
9. Load or create a template with at least three sections.
10. Save the template.
11. Generate one section.
12. Generate the full document.
13. Confirm that each section has a `Sources used` block.

---

## 13. License and Ownership

This repository is a proof-of-concept implementation for demonstrating a RAG-based document generation workflow.
