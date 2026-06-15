"""Prompt composition utilities for section-by-section document generation."""

from __future__ import annotations

from src.models import DocumentTemplate, RetrievedChunk, TemplateSection


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as source-aware context."""
    if not chunks:
        return "No retrieved context available."

    context_blocks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"File: {chunk.source_file}",
                    f"Page: {chunk.page_number}",
                    f"Chunk ID: {chunk.chunk_id}",
                    "Text:",
                    chunk.text,
                ]
            )
        )

    return "\n\n---\n\n".join(context_blocks)


def format_subsections(section: TemplateSection) -> str:
    """Format subsection titles and instructions."""
    if not section.subsections:
        return "No subsections."

    lines: list[str] = []

    for index, subsection in enumerate(section.subsections, start=1):
        lines.append(f"{index}. {subsection.title}")
        if subsection.instruction:
            lines.append(f"   Instruction: {subsection.instruction}")

    return "\n".join(lines)


def build_section_prompt(
    template: DocumentTemplate,
    section: TemplateSection,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    """
    Build the final LLM prompt for one document section.

    Args:
        template: Full user-defined document template.
        section: Current section to generate.
        retrieved_chunks: Context chunks retrieved for this section.

    Returns:
        Complete prompt string.
    """
    context = format_retrieved_context(retrieved_chunks)
    subsections = format_subsections(section)

    return f"""You are generating exactly ONE section of a structured document.

GLOBAL BASE PROMPT:
{template.base_prompt}

SECTION TO GENERATE:
{section.title}

SECTION-SPECIFIC INSTRUCTION:
{section.instruction or "No additional section-specific instruction."}

SUBSECTIONS TO INCLUDE:
{subsections}

RETRIEVED CONTEXT:
{context}

STRICT GENERATION RULES:
1. Generate ONLY the section titled "{section.title}".
2. Do NOT generate other major sections unless they are explicitly listed under SUBSECTIONS TO INCLUDE.
3. Do NOT create headings such as Education, Skills, Projects, Certificates, Methodology, Conclusion, or Professional Experience unless they are requested as this section title or subsection titles.
4. Use only facts supported by the retrieved context.
5. Do not invent or assume missing facts.
6. Keep the output in clean Markdown.
7. Write only the section body content. Do not include the main section heading.
8. If no subsections are provided, write only 1-3 focused paragraphs.
9. If subsections are provided, include only those subsection headings.
10. Do NOT write a "Sources used" section. The application will add citations separately.

TASK:
Write only the body content for the "{section.title}" section now.
"""
