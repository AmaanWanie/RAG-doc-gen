"""LLM client utilities for generating document sections."""

from __future__ import annotations

import re
from typing import Any, Protocol

import requests
from openai import OpenAI


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Final composed prompt.

        Returns:
            Generated text.
        """


class MockLLMClient:
    """Mock LLM client used for offline testing without any model."""

    def generate(self, prompt: str) -> str:
        """
        Return a deterministic mock section.

        Args:
            prompt: Final composed prompt.

        Returns:
            Mock generated section.
        """
        section_title = _extract_section_title(prompt)

        return (
            f"## {section_title}\n\n"
            "This is a mock generated section. The prompt builder, retrieval system, "
            "and generation pipeline are connected successfully.\n\n"
            "**Sources used:**\n"
            "- Sources are available in the retrieved context above.\n"
        )


class OllamaClient:
    """Local Ollama client using the Ollama REST API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 180,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            base_url: Ollama server base URL, usually http://localhost:11434.
            model: Local Ollama model name.
            timeout_seconds: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        """
        Generate text using a local Ollama model.

        Args:
            prompt: Final composed prompt.

        Returns:
            Generated text.
        """
        url = f"{self.base_url}/api/generate"

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running. "
                "If needed, run: ollama serve"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Ollama request timed out. Try Top-K=1 or use a smaller model."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        data = response.json()
        generated_text = data.get("response", "")

        if not generated_text:
            raise RuntimeError(
                "Ollama returned an empty response. Check that the model name is valid."
            )

        return str(generated_text).strip()


class OpenAIChatClient:
    """OpenAI chat completion client."""

    def __init__(self, api_key: str, model: str) -> None:
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key.
            model: OpenAI model name.
        """
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is not set. Add your real key to .env "
                "or use LLM_PROVIDER=ollama/mock."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Generate text from OpenAI.

        Args:
            prompt: Final composed prompt.

        Returns:
            Generated text.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate structured, factual document sections "
                        "using only provided context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content

        if not content:
            return ""

        return content.strip()


def _extract_section_title(prompt: str) -> str:
    """
    Extract the section title from a final prompt.

    Args:
        prompt: Final prompt.

    Returns:
        Section title if found, otherwise a fallback.
    """
    match = re.search(r"SECTION TITLE:\s*(.+?)\n", prompt)

    if match:
        return match.group(1).strip()

    return "Generated Section"


def get_llm_client(
    provider: str,
    openai_api_key: str,
    openai_model: str,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.2:latest",
) -> LLMClient:
    """
    Create an LLM client from provider settings.

    Args:
        provider: LLM provider name. Supported: mock, openai, ollama.
        openai_api_key: OpenAI API key.
        openai_model: OpenAI model name.
        ollama_base_url: Ollama local server URL.
        ollama_model: Ollama local model name.

    Returns:
        LLM client.
    """
    provider = provider.lower().strip()

    if provider == "mock":
        return MockLLMClient()

    if provider == "openai":
        return OpenAIChatClient(api_key=openai_api_key, model=openai_model)

    if provider == "ollama":
        return OllamaClient(base_url=ollama_base_url, model=ollama_model)

    raise ValueError(f"Unsupported LLM provider: {provider}")
