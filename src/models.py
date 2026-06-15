"""Core data models for the RAG Document Generator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Represents extracted text from one page or text document segment."""

    source_file: str = Field(..., description="Original uploaded file name.")
    file_path: str = Field(..., description="Saved local file path.")
    page_number: int = Field(..., description="Page number. Use 1 for plain text files.")
    text: str = Field(..., description="Extracted clean text.")


class ExtractionResult(BaseModel):
    """Represents the result of extracting text from one uploaded document."""

    source_file: str
    file_path: str
    pages: list[DocumentPage]
    total_characters: int


class DocumentChunk(BaseModel):
    """Represents a searchable chunk created from an extracted document page."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    source_file: str = Field(..., description="Original source file name.")
    file_path: str = Field(..., description="Saved local file path.")
    page_number: int = Field(..., description="Source page number.")
    chunk_index: int = Field(..., description="Chunk number within the page.")
    text: str = Field(..., description="Chunk text.")
    character_count: int = Field(..., description="Number of characters in the chunk.")


class ChunkingResult(BaseModel):
    """Represents chunks created from one extracted document."""

    source_file: str
    file_path: str
    chunks: list[DocumentChunk]
    total_chunks: int


class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved from the vector database."""

    chunk_id: str
    text: str
    source_file: str
    file_path: str
    page_number: int
    chunk_index: int
    distance: float | None = None


class TemplateSubsection(BaseModel):
    """Represents a subsection inside a generated document section."""

    title: str = Field(..., min_length=1)
    instruction: str = ""


class TemplateSection(BaseModel):
    """Represents one section in the user-defined generation template."""

    title: str = Field(..., min_length=1)
    instruction: str = ""
    subsections: list[TemplateSubsection] = Field(default_factory=list)


class DocumentTemplate(BaseModel):
    """Represents a full user-defined document generation template."""

    template_name: str = Field(..., min_length=1)
    base_prompt: str = Field(..., min_length=1)
    sections: list[TemplateSection] = Field(..., min_length=1)

class GeneratedSection(BaseModel):
    """Represents one generated document section with its citations."""

    title: str
    content: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    prompt: str

