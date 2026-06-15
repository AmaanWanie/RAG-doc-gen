"""Utilities for saving and loading user-defined prompt templates."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import TEMPLATE_DIR
from src.models import DocumentTemplate


def make_template_slug(template_name: str) -> str:
    """
    Convert a template name into a safe file name.

    Args:
        template_name: Human-readable template name.

    Returns:
        Safe lowercase file name stem.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", template_name).strip("_").lower()
    return slug or "template"


def get_template_path(template_name: str, template_dir: Path = TEMPLATE_DIR) -> Path:
    """
    Build the JSON path for a template.

    Args:
        template_name: Template name.
        template_dir: Directory where templates are stored.

    Returns:
        Path to the template JSON file.
    """
    return template_dir / f"{make_template_slug(template_name)}.json"


def save_template(
    template: DocumentTemplate,
    template_dir: Path = TEMPLATE_DIR,
) -> Path:
    """
    Save a document template as JSON.

    Args:
        template: Validated document template.
        template_dir: Directory where templates are stored.

    Returns:
        Path to the saved template file.
    """
    template_dir.mkdir(parents=True, exist_ok=True)
    output_path = get_template_path(template.template_name, template_dir)

    output_path.write_text(
        template.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path


def load_template(template_path: str | Path) -> DocumentTemplate:
    """
    Load a document template from JSON.

    Args:
        template_path: Path to the template JSON file.

    Returns:
        Validated DocumentTemplate object.
    """
    path = Path(template_path)

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return DocumentTemplate.model_validate(data)


def list_template_files(template_dir: Path = TEMPLATE_DIR) -> list[Path]:
    """
    List available saved template JSON files.

    Args:
        template_dir: Directory where templates are stored.

    Returns:
        Sorted list of template JSON files.
    """
    template_dir.mkdir(parents=True, exist_ok=True)
    return sorted(template_dir.glob("*.json"))
