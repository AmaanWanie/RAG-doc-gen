"""Generation pipeline for producing document sections."""

from __future__ import annotations

import re

from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import build_section_prompt
from src.models import DocumentTemplate, GeneratedSection, TemplateSection
from src.output.citation_formatter import format_sources_markdown
from src.retrieval.retriever import Retriever


def build_retrieval_query(section: TemplateSection) -> str:
    """Build a retrieval query from section title, instruction, and subsections."""
    query_parts = [section.title, section.instruction]

    for subsection in section.subsections:
        query_parts.append(subsection.title)
        query_parts.append(subsection.instruction)

    return " ".join(part for part in query_parts if part).strip()


def normalize_heading_text(text: str) -> str:
    """
    Normalize heading-like text for comparison.

    Args:
        text: Raw heading text.

    Returns:
        Lowercase normalized heading text.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"^\*\*|\*\*$", "", cleaned)
    cleaned = cleaned.strip(":").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def remove_model_generated_sources(text: str) -> str:
    """
    Remove any source/citation section produced by the model.

    Args:
        text: Raw generated text.

    Returns:
        Text without model-generated citation footer.
    """
    patterns = [
        r"\n\s*\*\*Sources used:\*\*[\s\S]*$",
        r"\n\s*Sources used:\s*[\s\S]*$",
        r"\n\s*#+\s*Sources used\s*[\s\S]*$",
    ]

    cleaned = text.strip()

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned


def remove_duplicate_section_heading(title: str, content: str) -> str:
    """
    Remove duplicate model-generated section title from the start of content.

    Handles examples:
        Introduction
        INTRODUCTION
        # Introduction
        ## Introduction
        **Introduction**

    Args:
        title: Expected section title.
        content: Raw section content.

    Returns:
        Content without duplicate leading section heading.
    """
    lines = content.strip().splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    if not lines:
        return ""

    first_line = lines[0].strip()

    if normalize_heading_text(first_line) == normalize_heading_text(title):
        lines.pop(0)

        while lines and not lines[0].strip():
            lines.pop(0)

    return "\n".join(lines).strip()


def demote_unwanted_top_level_headings(content: str) -> str:
    """
    Convert any model-generated top-level headings to second-level headings.

    Args:
        content: Generated Markdown content.

    Returns:
        Markdown content without accidental # top-level headings.
    """
    lines = content.splitlines()
    fixed_lines: list[str] = []

    for line in lines:
        if re.match(r"^#(?!#)\s+", line):
            fixed_lines.append("#" + line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines).strip()


def ensure_section_heading(title: str, content: str) -> str:
    """
    Force generated content to start with exactly one level-2 Markdown heading.

    Args:
        title: Section title.
        content: Generated section body.

    Returns:
        Markdown content with a clean section heading.
    """
    cleaned = remove_duplicate_section_heading(title, content)
    cleaned = demote_unwanted_top_level_headings(cleaned)

    if not cleaned:
        cleaned = "No supported content was generated for this section."

    return f"## {title}\n\n{cleaned}"


def generate_section(
    template: DocumentTemplate,
    section: TemplateSection,
    retriever: Retriever,
    llm_client: LLMClient,
    top_k: int = 3,
    max_distance: float | str | None = "auto",
) -> GeneratedSection:
    """
    Generate one document section.

    Args:
        template: Full document template.
        section: Section to generate.
        retriever: Retriever for fetching relevant context.
        llm_client: LLM client used for generation.
        top_k: Number of chunks to retrieve before filtering.
        max_distance: Optional distance threshold or auto filtering mode.

    Returns:
        GeneratedSection with output, prompt, and retrieved chunks.
    """
    retrieval_query = build_retrieval_query(section)

    retrieved_chunks = retriever.retrieve(
        query=retrieval_query,
        top_k=top_k,
        max_distance=max_distance,
    )

    prompt = build_section_prompt(
        template=template,
        section=section,
        retrieved_chunks=retrieved_chunks,
    )

    raw_content = llm_client.generate(prompt)
    body_content = remove_model_generated_sources(raw_content)
    section_content = ensure_section_heading(section.title, body_content)

    final_content = (
        f"{section_content}\n\n"
        f"{format_sources_markdown(retrieved_chunks)}"
    )

    retrieved_chunk_payloads = [
        chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
        for chunk in retrieved_chunks
    ]

    return GeneratedSection(
        title=section.title,
        content=final_content,
        retrieved_chunks=retrieved_chunk_payloads,
        prompt=prompt,
    )


def generate_document(
    template: DocumentTemplate,
    retriever: Retriever,
    llm_client: LLMClient,
    top_k: int = 3,
    max_distance: float | str | None = "auto",
) -> list[GeneratedSection]:
    """
    Generate all sections in a document template.

    Args:
        template: Full document template.
        retriever: Retriever for fetching relevant context.
        llm_client: LLM client used for generation.
        top_k: Number of chunks to retrieve before filtering.
        max_distance: Optional distance threshold or auto filtering mode.

    Returns:
        List of generated sections in template order.
    """
    generated_sections: list[GeneratedSection] = []

    for section in template.sections:
        generated_section = generate_section(
            template=template,
            section=section,
            retriever=retriever,
            llm_client=llm_client,
            top_k=top_k,
            max_distance=max_distance,
        )
        generated_sections.append(generated_section)

    return generated_sections


def assemble_document_markdown(
    template: DocumentTemplate,
    generated_sections: list[GeneratedSection],
) -> str:
    """
    Assemble generated sections into one Markdown document.

    Args:
        template: Document template.
        generated_sections: Generated sections.

    Returns:
        Full Markdown document.
    """
    parts = [f"# {template.template_name}"]

    for section in generated_sections:
        parts.append(section.content.strip())

    return "\n\n---\n\n".join(parts).strip() + "\n"
