"""Streamlit UI for the RAG Document Generator POC."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config import (
    CHROMA_DB_DIR,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OUTPUT_DIR,
    TEMPLATE_DIR,
    UPLOAD_DIR,
    ensure_directories,
)
from src.generation.generation_pipeline import (
    assemble_document_markdown,
    generate_section,
)
from src.generation.llm_client import get_llm_client
from src.generation.prompt_builder import build_section_prompt
from src.ingestion.document_loader import load_document, save_uploaded_file
from src.ingestion.ingestion_pipeline import create_chunks_from_extraction
from src.models import (
    DocumentChunk,
    DocumentTemplate,
    TemplateSection,
    TemplateSubsection,
)
from src.retrieval.retriever import Retriever
from src.templates.template_manager import (
    list_template_files,
    load_template,
    save_template,
)
from src.vectorstore.chroma_store import ChromaVectorStore
from src.vectorstore.embeddings import LocalEmbeddingModel


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> LocalEmbeddingModel:
    """Load and cache the local embedding model."""
    return LocalEmbeddingModel()


@st.cache_resource(show_spinner=False)
def get_vector_store() -> ChromaVectorStore:
    """Load and cache the ChromaDB vector store."""
    return ChromaVectorStore(persist_directory=CHROMA_DB_DIR)


def apply_custom_styles() -> None:
    """Apply lightweight custom CSS for a cleaner Streamlit UI."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        h1 {
            font-size: 2.35rem !important;
            margin-bottom: 0.25rem !important;
        }

        h2, h3 {
            margin-top: 1.5rem !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 0.75rem;
        }

        div[data-testid="stExpander"] {
            border-radius: 0.75rem;
        }

        .workflow-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(250, 250, 250, 0.12);
            border-radius: 0.85rem;
            background: rgba(255, 255, 255, 0.03);
            margin-bottom: 1rem;
        }

        .workflow-title {
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.25rem;
        }

        .workflow-text {
            color: #cbd5e1;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_guide() -> None:
    """Render a simple sidebar workflow guide."""
    st.sidebar.title("RAG Document Generator")

    st.sidebar.markdown(
        """
        **Workflow**

        1. Upload + chunk documents  
        2. Build retrieval index  
        3. Configure template  
        4. Preview prompt  
        5. Generate one section  
        6. Generate full document  
        """
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        **Recommended settings**

        - Chunk size: `1000`
        - Chunk overlap: `200`
        - Retrieval: `Auto distance`
        - Provider: `Ollama`
        - Top-K: `2`
        """
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "Tip: Use Top-K 1–2 for small documents and 3–5 for larger reports."
    )


def render_intro_panel() -> None:
    """Render a friendly introduction panel."""
    st.markdown(
        """
        <div class="workflow-card">
            <div class="workflow-title">Generate structured documents from uploaded sources</div>
            <div class="workflow-text">
                Upload documents, define a reusable prompt template, retrieve relevant source chunks,
                and generate a cited Markdown document section by section.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("How to use this app", expanded=False):
        st.markdown(
            """
            **Quick test flow**

            1. Upload a PDF or text file.
            2. Choose chunk size and overlap.
            3. Click **Upload, Extract and Create Chunks**.
            4. Review extracted text or chunks if needed.
            5. Click **Create / Reset Vector Index**.
            6. Load or create a prompt template.
            7. Generate one section first.
            8. Generate the full document.
            """
        )


def initialize_runtime_state() -> None:
    """Initialize runtime session state variables."""
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = []

    if "chunking_results" not in st.session_state:
        st.session_state.chunking_results = []

    if "vector_index_ready" not in st.session_state:
        st.session_state.vector_index_ready = False


def flatten_chunks_from_session() -> list[DocumentChunk]:
    """Collect all chunks currently stored in Streamlit session state."""
    all_chunks: list[DocumentChunk] = []

    for result in st.session_state.get("chunking_results", []):
        all_chunks.extend(result.chunks)

    return all_chunks


def save_generated_markdown(template_name: str, markdown: str) -> Path:
    """Save generated Markdown output to the outputs directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(
        char.lower() if char.isalnum() else "_"
        for char in template_name.strip()
    ).strip("_") or "generated_document"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{safe_name}_{timestamp}.md"
    output_path.write_text(markdown, encoding="utf-8")

    return output_path


@st.dialog("Review extracted text")
def render_extraction_review_dialog() -> None:
    """Show extracted text in a dialog."""
    results = st.session_state.get("extraction_results", [])

    if not results:
        st.info("No extracted text is available yet. Upload and process documents first.")
        return

    for result in results:
        with st.expander(
            f"{result.source_file} — {len(result.pages)} page(s)",
            expanded=True,
        ):
            st.write(f"Saved path: `{result.file_path}`")
            st.write(f"Total characters: `{result.total_characters:,}`")

            for page in result.pages:
                st.markdown(f"**Page {page.page_number}**")

                show_full_text = st.checkbox(
                    f"Show full text for page {page.page_number}",
                    value=False,
                    key=f"dialog_show_full_{result.source_file}_{page.page_number}",
                )

                preview_text = page.text if show_full_text else page.text[:1500]

                st.text_area(
                    label=f"{page.source_file} — page {page.page_number}",
                    value=preview_text,
                    height=360 if show_full_text else 190,
                    disabled=True,
                )

                if not show_full_text and len(page.text) > 1500:
                    st.info(
                        f"Showing first 1,500 characters only. "
                        f"Full page has {len(page.text):,} characters."
                    )


@st.dialog("Review chunks")
def render_chunk_review_dialog() -> None:
    """Show generated chunks in a dialog."""
    results = st.session_state.get("chunking_results", [])

    if not results:
        st.info("No chunks are available yet. Upload, extract, and create chunks first.")
        return

    total_chunks = sum(result.total_chunks for result in results)
    st.success(f"Total chunks created: {total_chunks}")

    for result in results:
        with st.expander(
            f"{result.source_file} — {result.total_chunks} chunk(s)",
            expanded=True,
        ):
            for chunk in result.chunks:
                st.markdown(
                    f"**{chunk.chunk_id}** | Page {chunk.page_number} | "
                    f"{chunk.character_count} characters"
                )
                st.text_area(
                    label=f"Chunk text: {chunk.chunk_id}",
                    value=chunk.text,
                    height=170,
                    disabled=True,
                )


@st.dialog("Test retrieval")
def render_retrieval_test_dialog() -> None:
    """Show retrieval testing controls in a dialog."""
    if not st.session_state.get("vector_index_ready", False):
        st.info("Build the vector index first before testing retrieval.")
        return

    query = st.text_input(
        "Search query",
        value="system architecture",
        help="Try topics like system architecture, evaluation metrics, privacy risks.",
        key="dialog_retrieval_query",
    )

    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=10,
        value=3,
        key="dialog_retrieval_top_k",
    )

    use_auto_distance = st.checkbox(
        "Use automatic distance filter",
        value=True,
        key="dialog_retrieval_auto_distance",
    )

    manual_distance = st.slider(
        "Manual maximum distance filter",
        min_value=0.10,
        max_value=1.50,
        value=0.65,
        step=0.05,
        key="dialog_retrieval_manual_distance",
        disabled=use_auto_distance,
    )

    distance_filter = "auto" if use_auto_distance else manual_distance

    if st.button("Retrieve Relevant Chunks", type="primary"):
        try:
            with st.spinner("Retrieving relevant chunks..."):
                retriever = Retriever(
                    embedding_model=get_embedding_model(),
                    vector_store=get_vector_store(),
                )
                retrieved_chunks = retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    max_distance=distance_filter,
                )

            st.session_state.dialog_retrieval_results = retrieved_chunks

        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")

    retrieved_chunks = st.session_state.get("dialog_retrieval_results", [])

    if retrieved_chunks:
        st.success(f"Retrieved {len(retrieved_chunks)} chunk(s).")

        for chunk in retrieved_chunks:
            title = (
                f"{chunk.source_file} | Page {chunk.page_number} | "
                f"Chunk {chunk.chunk_index}"
            )

            if chunk.distance is not None:
                title += f" | Distance: {chunk.distance:.4f}"

            with st.expander(title, expanded=True):
                st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
                st.text_area(
                    label=f"Retrieved text: {chunk.chunk_id}",
                    value=chunk.text,
                    height=180,
                    disabled=True,
                )


def render_document_ingestion_step() -> None:
    """Render upload, extraction, chunking, and review buttons together."""
    st.header("Step 1 — Upload, extract, and chunk source documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        chunk_size = st.number_input(
            "Chunk size",
            min_value=300,
            max_value=3000,
            value=1000,
            step=100,
            help="Maximum characters per chunk.",
        )

    with col2:
        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=1000,
            value=200,
            step=50,
            help="Approximate overlap between adjacent chunks.",
        )

    if st.button("Upload, Extract and Create Chunks", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
            return

        st.session_state.extraction_results = []
        st.session_state.chunking_results = []
        st.session_state.vector_index_ready = False
        st.session_state.pop("dialog_retrieval_results", None)

        with st.spinner("Saving, extracting, and chunking documents..."):
            for uploaded_file in uploaded_files:
                try:
                    saved_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
                    extraction = load_document(saved_path)
                    st.session_state.extraction_results.append(extraction)

                    chunking_result = create_chunks_from_extraction(
                        extraction=extraction,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                    st.session_state.chunking_results.append(chunking_result)

                    st.success(
                        f"Processed {extraction.source_file}: "
                        f"{len(extraction.pages)} page(s), "
                        f"{chunking_result.total_chunks} chunk(s)."
                    )

                except Exception as exc:
                    st.error(f"Failed to process {uploaded_file.name}: {exc}")

    if st.session_state.extraction_results:
        total_pages = sum(
            len(result.pages) for result in st.session_state.extraction_results
        )
        total_chunks = sum(
            result.total_chunks for result in st.session_state.chunking_results
        )
        total_characters = sum(
            result.total_characters for result in st.session_state.extraction_results
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Documents", len(st.session_state.extraction_results))
        metric_col2.metric("Pages", total_pages)
        metric_col3.metric("Chunks", total_chunks)

        st.caption(f"Extracted {total_characters:,} total characters.")

        review_col1, review_col2 = st.columns(2)

        with review_col1:
            if st.button("Review Extracted Text"):
                render_extraction_review_dialog()

        with review_col2:
            if st.button("Review Chunks"):
                render_chunk_review_dialog()


def render_vector_index_step() -> None:
    """Render vector indexing and retrieval test button."""
    st.header("Step 5 — Build retrieval index")

    all_chunks = flatten_chunks_from_session()

    if not all_chunks:
        st.info("Upload and process documents first. Chunks will appear here after Step 1.")
        return

    st.write(f"Ready to index `{len(all_chunks)}` chunks into ChromaDB.")
    st.write(f"ChromaDB directory: `{CHROMA_DB_DIR}`")

    if st.button("Create / Reset Vector Index", type="primary"):
        try:
            with st.spinner(
                "Loading embedding model and indexing chunks. First run may take longer..."
            ):
                embedding_model = get_embedding_model()
                vector_store = get_vector_store()

                vector_store.reset_collection()

                texts = [chunk.text for chunk in all_chunks]
                embeddings = embedding_model.embed_texts(texts)

                inserted_count = vector_store.add_chunks(
                    chunks=all_chunks,
                    embeddings=embeddings,
                )

            st.session_state.vector_index_ready = True
            st.success(
                f"Vector index created successfully. Stored {inserted_count} chunk(s)."
            )

        except Exception as exc:
            st.session_state.vector_index_ready = False
            st.error(f"Vector indexing failed: {exc}")

    if st.session_state.get("vector_index_ready", False):
        st.success("Retrieval index is ready.")

        if st.button("Test Retrieval"):
            render_retrieval_test_dialog()


def initialize_template_state() -> None:
    """Initialize template editor state."""
    if "template_editor_version" not in st.session_state:
        st.session_state.template_editor_version = 0

    if "template_name" not in st.session_state:
        st.session_state.template_name = "Smart Classroom AI Report"

    if "base_prompt" not in st.session_state:
        st.session_state.base_prompt = (
            "Write in a professional technical report style. "
            "Use only the retrieved context. Do not invent facts. "
            "Keep the output clear and structured."
        )

    if "template_sections" not in st.session_state:
        st.session_state.template_sections = [
            {
                "title": "Introduction",
                "instruction": "Introduce the topic and explain its background.",
                "subsections": [],
            },
            {
                "title": "Executive Summary",
                "instruction": "Summarize the purpose and value of the system.",
                "subsections": [],
            },
            {
                "title": "System Architecture",
                "instruction": "Explain the main components and how they work together.",
                "subsections": [
                    {
                        "title": "Core Modules",
                        "instruction": (
                            "Describe ingestion, retrieval, prompt building, "
                            "generation, and output rendering."
                        ),
                    }
                ],
            },
        ]


def clear_template_widget_state() -> None:
    """Clear stale Streamlit widget keys used by the template editor."""
    prefixes = (
        "v",
        "section_title_",
        "section_instruction_",
        "subsection_title_",
        "subsection_instruction_",
        "add_subsection_",
        "remove_section_",
        "remove_subsection_",
    )

    for key in list(st.session_state.keys()):
        if key.startswith(prefixes):
            if key not in {"vector_index_ready"}:
                del st.session_state[key]


def build_template_from_state() -> DocumentTemplate:
    """Build and validate a DocumentTemplate from Streamlit session state."""
    sections: list[TemplateSection] = []

    for section in st.session_state.template_sections:
        section_title = section.get("title", "").strip()
        section_instruction = section.get("instruction", "").strip()

        if not section_title:
            continue

        subsections: list[TemplateSubsection] = []

        for subsection in section.get("subsections", []):
            subsection_title = subsection.get("title", "").strip()
            subsection_instruction = subsection.get("instruction", "").strip()

            if subsection_title:
                subsections.append(
                    TemplateSubsection(
                        title=subsection_title,
                        instruction=subsection_instruction,
                    )
                )

        sections.append(
            TemplateSection(
                title=section_title,
                instruction=section_instruction,
                subsections=subsections,
            )
        )

    return DocumentTemplate(
        template_name=st.session_state.template_name.strip(),
        base_prompt=st.session_state.base_prompt.strip(),
        sections=sections,
    )


def load_template_into_state(template: DocumentTemplate) -> None:
    """Load a template object into Streamlit session state."""
    clear_template_widget_state()

    st.session_state.template_editor_version = (
        st.session_state.get("template_editor_version", 0) + 1
    )

    st.session_state.template_name = template.template_name
    st.session_state.base_prompt = template.base_prompt
    st.session_state.template_sections = [
        {
            "title": section.title,
            "instruction": section.instruction,
            "subsections": [
                {
                    "title": subsection.title,
                    "instruction": subsection.instruction,
                }
                for subsection in section.subsections
            ],
        }
        for section in template.sections
    ]


def render_template_editor() -> None:
    """Render the prompt template editor UI."""
    initialize_template_state()

    st.header("Step 7 — Configure prompt template")

    editor_version = st.session_state.get("template_editor_version", 0)

    st.subheader("Load Existing Template")

    template_files = list_template_files(TEMPLATE_DIR)
    template_options = [str(path) for path in template_files]

    if template_options:
        selected_template_path = st.selectbox(
            "Saved templates",
            options=template_options,
            format_func=lambda path: Path(path).name,
            placeholder="Select a saved template",
            key="selected_template_path",
        )

        if st.button("Load Selected Template"):
            try:
                template = load_template(selected_template_path)
                load_template_into_state(template)
                st.success(f"Loaded template: {template.template_name}")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load template: {exc}")
    else:
        st.info("No saved templates found yet. Create and save one below.")

    st.divider()

    st.subheader("Edit Template")

    st.text_input("Template name", key="template_name")

    st.text_area(
        "Base prompt",
        key="base_prompt",
        height=140,
        help="This global instruction will be applied to every generated section.",
    )

    st.subheader("Sections")

    if st.button("Add Section"):
        st.session_state.template_sections.append(
            {
                "title": "",
                "instruction": "",
                "subsections": [],
            }
        )
        st.rerun()

    for section_index, section in enumerate(st.session_state.template_sections):
        with st.expander(
            f"Section {section_index + 1}: {section.get('title') or 'Untitled'}",
            expanded=True,
        ):
            section["title"] = st.text_input(
                "Section title",
                value=section.get("title", ""),
                key=f"v{editor_version}_section_title_{section_index}",
            )

            section["instruction"] = st.text_area(
                "Section-specific instruction",
                value=section.get("instruction", ""),
                height=100,
                key=f"v{editor_version}_section_instruction_{section_index}",
            )

            st.markdown("**Subsections**")

            sub_add_col, sec_remove_col = st.columns(2)

            with sub_add_col:
                if st.button(
                    "Add Subsection",
                    key=f"v{editor_version}_add_subsection_{section_index}",
                ):
                    section["subsections"].append(
                        {
                            "title": "",
                            "instruction": "",
                        }
                    )
                    st.rerun()

            with sec_remove_col:
                if st.button(
                    "Remove Section",
                    key=f"v{editor_version}_remove_section_{section_index}",
                ):
                    st.session_state.template_sections.pop(section_index)
                    st.rerun()

            for subsection_index, subsection in enumerate(section["subsections"]):
                st.markdown(f"Subsection {subsection_index + 1}")

                subsection["title"] = st.text_input(
                    "Subsection title",
                    value=subsection.get("title", ""),
                    key=(
                        f"v{editor_version}_subsection_title_"
                        f"{section_index}_{subsection_index}"
                    ),
                )

                subsection["instruction"] = st.text_area(
                    "Subsection instruction",
                    value=subsection.get("instruction", ""),
                    height=80,
                    key=(
                        f"v{editor_version}_subsection_instruction_"
                        f"{section_index}_{subsection_index}"
                    ),
                )

                if st.button(
                    "Remove Subsection",
                    key=(
                        f"v{editor_version}_remove_subsection_"
                        f"{section_index}_{subsection_index}"
                    ),
                ):
                    section["subsections"].pop(subsection_index)
                    st.rerun()

    save_col, preview_col = st.columns(2)

    with save_col:
        if st.button("Save Template", type="primary"):
            try:
                template = build_template_from_state()
                output_path = save_template(template)
                st.success(f"Template saved: {output_path}")
            except Exception as exc:
                st.error(f"Template save failed: {exc}")

    with preview_col:
        if st.button("Validate Template"):
            try:
                template = build_template_from_state()
                st.success(
                    f"Template valid: {len(template.sections)} section(s) configured."
                )
            except Exception as exc:
                st.error(f"Template validation failed: {exc}")

    with st.expander("Advanced: template JSON preview", expanded=False):
        try:
            template = build_template_from_state()
            st.json(json.loads(template.model_dump_json()))
        except Exception as exc:
            st.warning(f"Template is not valid yet: {exc}")


def render_prompt_preview() -> None:
    """Render a preview of the final prompt for one selected section."""
    st.header("Step 8 — Preview final prompt")

    try:
        template = build_template_from_state()
    except Exception as exc:
        st.warning(f"Create a valid template before previewing prompts: {exc}")
        return

    if not st.session_state.get("vector_index_ready", False):
        st.info("Build the vector index first to preview retrieved-context prompts.")
        return

    section_titles = [section.title for section in template.sections]

    selected_section_title = st.selectbox(
        "Select section for prompt preview",
        options=section_titles,
        key="prompt_preview_section",
    )

    selected_section = next(
        section for section in template.sections if section.title == selected_section_title
    )

    top_k_prompt = st.slider(
        "Top-K context chunks for prompt",
        min_value=1,
        max_value=10,
        value=3,
        key="prompt_preview_top_k",
    )

    use_auto_distance_prompt = st.checkbox(
        "Use automatic distance filter for prompt context",
        value=True,
        key="prompt_preview_use_auto_distance",
    )

    max_distance_prompt = st.slider(
        "Manual maximum distance filter for prompt context",
        min_value=0.10,
        max_value=1.50,
        value=0.65,
        step=0.05,
        key="prompt_preview_max_distance",
        disabled=use_auto_distance_prompt,
    )

    distance_filter_prompt = "auto" if use_auto_distance_prompt else max_distance_prompt

    if st.button("Build Prompt Preview"):
        try:
            with st.spinner("Retrieving context and building prompt..."):
                retriever = Retriever(
                    embedding_model=get_embedding_model(),
                    vector_store=get_vector_store(),
                )

                retrieval_query = selected_section.title
                if selected_section.instruction:
                    retrieval_query += " " + selected_section.instruction

                retrieved_chunks = retriever.retrieve(
                    query=retrieval_query,
                    top_k=top_k_prompt,
                    max_distance=distance_filter_prompt,
                )

                final_prompt = build_section_prompt(
                    template=template,
                    section=selected_section,
                    retrieved_chunks=retrieved_chunks,
                )

            st.success(
                f"Prompt built for section: {selected_section.title}. "
                f"Retrieved {len(retrieved_chunks)} chunk(s)."
            )

            with st.expander("Retrieved chunks used for this prompt", expanded=True):
                for chunk in retrieved_chunks:
                    st.markdown(
                        f"**{chunk.source_file} | Page {chunk.page_number} | "
                        f"Chunk {chunk.chunk_index}**"
                    )
                    st.text_area(
                        label=f"Context chunk: {chunk.chunk_id}",
                        value=chunk.text,
                        height=140,
                        disabled=True,
                    )

            st.text_area(
                "Final prompt preview",
                value=final_prompt,
                height=500,
                disabled=True,
            )

        except Exception as exc:
            st.error(f"Prompt preview failed: {exc}")


def render_single_section_generation() -> None:
    """Render UI for generating one selected section."""
    st.header("Step 9 — Generate one section")

    try:
        template = build_template_from_state()
    except Exception as exc:
        st.warning(f"Create a valid template before generation: {exc}")
        return

    if not st.session_state.get("vector_index_ready", False):
        st.info("Build the vector index first before generating sections.")
        return

    section_titles = [section.title for section in template.sections]

    selected_section_title = st.selectbox(
        "Select section to generate",
        options=section_titles,
        key="generation_section",
    )

    selected_section = next(
        section for section in template.sections if section.title == selected_section_title
    )

    provider_options = ["ollama", "openai", "mock"]
    default_provider = LLM_PROVIDER if LLM_PROVIDER in provider_options else "ollama"

    selected_provider = st.selectbox(
        "LLM provider",
        options=provider_options,
        index=provider_options.index(default_provider),
        key="generation_provider",
    )

    col1, col2 = st.columns(2)

    with col1:
        generation_top_k = st.slider(
            "Top-K chunks for generation",
            min_value=1,
            max_value=10,
            value=2,
            key="generation_top_k",
        )

    with col2:
        use_auto_distance_generation = st.checkbox(
            "Use automatic distance filter for generation",
            value=True,
            key="generation_use_auto_distance",
        )

        generation_max_distance = st.slider(
            "Manual maximum distance filter for generation",
            min_value=0.10,
            max_value=1.50,
            value=0.65,
            step=0.05,
            key="generation_max_distance",
            disabled=use_auto_distance_generation,
        )

    distance_filter_generation = (
        "auto" if use_auto_distance_generation else generation_max_distance
    )

    if selected_provider == "ollama":
        st.info(
            "Ollama mode uses your local model. "
            f"Server: {OLLAMA_BASE_URL} | Model: {OLLAMA_MODEL}"
        )
    elif selected_provider == "openai":
        st.info(
            "OpenAI mode requires a real OPENAI_API_KEY in your .env file. "
            f"Current model: {OPENAI_MODEL}"
        )
    else:
        st.info("Mock mode tests the pipeline without calling a real model.")

    if st.button("Generate Selected Section", type="primary"):
        try:
            with st.spinner("Generating selected section..."):
                retriever = Retriever(
                    embedding_model=get_embedding_model(),
                    vector_store=get_vector_store(),
                )

                llm_client = get_llm_client(
                    provider=selected_provider,
                    openai_api_key=OPENAI_API_KEY,
                    openai_model=OPENAI_MODEL,
                    ollama_base_url=OLLAMA_BASE_URL,
                    ollama_model=OLLAMA_MODEL,
                )

                generated_section = generate_section(
                    template=template,
                    section=selected_section,
                    retriever=retriever,
                    llm_client=llm_client,
                    top_k=generation_top_k,
                    max_distance=distance_filter_generation,
                )

            st.session_state.generated_section = generated_section
            st.success(f"Generated section: {generated_section.title}")

        except Exception as exc:
            st.error(f"Section generation failed: {exc}")

    if "generated_section" in st.session_state:
        generated_section = st.session_state.generated_section

        st.subheader("Generated Output")
        st.markdown(generated_section.content)

        with st.expander("Advanced: retrieved chunks used", expanded=False):
            for chunk in generated_section.retrieved_chunks:
                title = (
                    f"{chunk.source_file} | Page {chunk.page_number} | "
                    f"Chunk {chunk.chunk_index}"
                )

                if chunk.distance is not None:
                    title += f" | Distance: {chunk.distance:.4f}"

                st.markdown(f"**{title}**")
                st.text_area(
                    label=f"Generated context chunk: {chunk.chunk_id}",
                    value=chunk.text,
                    height=140,
                    disabled=True,
                )

        with st.expander("Advanced: final prompt sent to generator", expanded=False):
            st.text_area(
                "Prompt",
                value=generated_section.prompt,
                height=400,
                disabled=True,
            )


def render_full_document_generation() -> None:
    """Render UI for generating the full document section by section."""
    st.header("Step 10 — Generate full document")

    try:
        template = build_template_from_state()
    except Exception as exc:
        st.warning(f"Create a valid template before full document generation: {exc}")
        return

    if not st.session_state.get("vector_index_ready", False):
        st.info("Build the vector index first before generating the full document.")
        return

    provider_options = ["ollama", "openai", "mock"]
    default_provider = LLM_PROVIDER if LLM_PROVIDER in provider_options else "ollama"

    selected_provider = st.selectbox(
        "Full document LLM provider",
        options=provider_options,
        index=provider_options.index(default_provider),
        key="full_generation_provider",
    )

    col1, col2 = st.columns(2)

    with col1:
        full_top_k = st.slider(
            "Top-K chunks per section",
            min_value=1,
            max_value=10,
            value=2,
            key="full_generation_top_k",
        )

    with col2:
        use_auto_distance_full = st.checkbox(
            "Use automatic distance filter per section",
            value=True,
            key="full_generation_use_auto_distance",
        )

        full_max_distance = st.slider(
            "Manual maximum distance filter per section",
            min_value=0.10,
            max_value=1.50,
            value=0.65,
            step=0.05,
            key="full_generation_max_distance",
            disabled=use_auto_distance_full,
        )

    distance_filter_full = "auto" if use_auto_distance_full else full_max_distance

    st.write(f"Template sections to generate: `{len(template.sections)}`")

    if selected_provider == "ollama":
        st.info(
            "Ollama mode uses your local model. "
            f"Server: {OLLAMA_BASE_URL} | Model: {OLLAMA_MODEL}"
        )
    elif selected_provider == "openai":
        st.info(
            "OpenAI mode requires a real OPENAI_API_KEY in your .env file. "
            f"Current model: {OPENAI_MODEL}"
        )
    else:
        st.info("Mock mode tests the full pipeline without calling a real model.")

    if st.button("Generate Full Document", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_box = st.empty()

            retriever = Retriever(
                embedding_model=get_embedding_model(),
                vector_store=get_vector_store(),
            )

            llm_client = get_llm_client(
                provider=selected_provider,
                openai_api_key=OPENAI_API_KEY,
                openai_model=OPENAI_MODEL,
                ollama_base_url=OLLAMA_BASE_URL,
                ollama_model=OLLAMA_MODEL,
            )

            generated_sections = []

            for index, section in enumerate(template.sections, start=1):
                status_box.info(
                    f"Generating section {index}/{len(template.sections)}: "
                    f"{section.title}"
                )

                generated_section = generate_section(
                    template=template,
                    section=section,
                    retriever=retriever,
                    llm_client=llm_client,
                    top_k=full_top_k,
                    max_distance=distance_filter_full,
                )

                generated_sections.append(generated_section)
                progress_bar.progress(index / len(template.sections))

            document_markdown = assemble_document_markdown(
                template=template,
                generated_sections=generated_sections,
            )

            output_path = save_generated_markdown(
                template_name=template.template_name,
                markdown=document_markdown,
            )

            st.session_state.generated_sections = generated_sections
            st.session_state.generated_document_markdown = document_markdown
            st.session_state.generated_document_path = str(output_path)

            status_box.success("Full document generated successfully.")

        except Exception as exc:
            st.error(f"Full document generation failed: {exc}")

    if "generated_document_markdown" in st.session_state:
        st.subheader("Generated Full Document")

        if "generated_document_path" in st.session_state:
            st.success(f"Saved file: `{st.session_state.generated_document_path}`")

        st.markdown(st.session_state.generated_document_markdown)

        st.download_button(
            label="Download Markdown",
            data=st.session_state.generated_document_markdown,
            file_name=f"{template.template_name.replace(' ', '_').lower()}.md",
            mime="text/markdown",
        )

        with st.expander("Advanced: generated sections debug view", expanded=False):
            for section in st.session_state.get("generated_sections", []):
                st.markdown(f"### {section.title}")
                st.write(f"Retrieved chunks: `{len(section.retrieved_chunks)}`")
                st.text_area(
                    label=f"Prompt for {section.title}",
                    value=section.prompt,
                    height=250,
                    disabled=True,
                )


def main() -> None:
    """Render the main Streamlit application."""
    ensure_directories()
    initialize_runtime_state()

    st.set_page_config(
        page_title="RAG Document Generator",
        page_icon="📄",
        layout="wide",
    )

    apply_custom_styles()
    render_sidebar_guide()

    st.title("📄 RAG Document Generator")
    st.caption(
        "Upload documents, configure a template, and generate a cited Markdown report."
    )
    render_intro_panel()

    render_document_ingestion_step()
    render_vector_index_step()
    render_template_editor()
    render_prompt_preview()
    render_single_section_generation()
    render_full_document_generation()


if __name__ == "__main__":
    main()
