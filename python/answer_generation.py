"""Provider-independent answer generation over a RAG context package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from context_builder import ContextPackage


class AnswerGenerator(Protocol):
    """The small interface any answer model needs to implement."""

    provider_name: str
    model_name: str

    def generate(self, context: ContextPackage) -> str:
        ...


class AnswerGenerationError(RuntimeError):
    """Raised when an answer provider cannot complete a request."""


@dataclass(frozen=True)
class AnswerRequest:
    question: str
    evidence: str


def build_grounded_prompt(context: ContextPackage) -> str:
    """Create a strict prompt that keeps the model grounded in retrieved text."""
    return (
        "Answer the question using only the evidence below. "
        "If the evidence is insufficient, say so clearly. "
        "Cite supporting evidence with its bracket citation, such as [1].\n\n"
        f"Question:\n{context.query}\n\n"
        f"Evidence:\n{context.context_text}"
    )


class MockAnswerGenerator:
    """Deterministic provider used by the UI and tests without an LLM download."""

    provider_name = "mock"
    model_name = "deterministic-demo"

    def generate(self, context: ContextPackage) -> str:
        if not context.evidence:
            return "I do not have enough retrieved evidence to answer this question."
        citations = " ".join(item["citation"] for item in context.evidence)
        first_sentence = context.evidence[0]["text"].strip()
        return (
            f"Based on the retrieved evidence, {first_sentence} "
            f"(sources: {citations})."
        )


class OllamaAnswerGenerator:
    """Optional connector for a locally running Ollama HTTP server."""

    provider_name = "ollama"

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, context: ContextPackage) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": build_grounded_prompt(context),
                "stream": False,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise AnswerGenerationError(
                f"Ollama returned HTTP {error.code}. Is model '{self.model_name}' available?"
            ) from error
        except URLError as error:
            raise AnswerGenerationError(
                "Could not reach Ollama at "
                f"{self.base_url}. Start Ollama or choose the mock generator."
            ) from error
        except (TimeoutError, json.JSONDecodeError) as error:
            raise AnswerGenerationError("Ollama returned an invalid or timed-out response") from error

        answer = body.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise AnswerGenerationError("Ollama response did not contain answer text")
        return answer.strip()


def generate_grounded_answer(context: ContextPackage, generator: AnswerGenerator) -> str:
    """Generate an answer through a selected provider."""
    if not context.query.strip():
        raise ValueError("context query cannot be empty")
    return generator.generate(context)

