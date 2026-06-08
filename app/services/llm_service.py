"""LLM-backed resume information extraction."""

from __future__ import annotations

import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import ValidationError

from app.models import ResumeData
from app.utils.logger import get_logger

logger = get_logger(__name__)

class LLMConfigurationError(RuntimeError):
    """Raised when no usable LLM provider has been configured."""


class LLMExtractionError(RuntimeError):
    """Raised when the LLM response cannot be used."""


class LLMRateLimitError(LLMExtractionError):
    """Raised when the provider returns a rate-limit (429) response."""


class LLMServiceError(LLMExtractionError):
    """Raised on transient 5xx / server-side errors from the provider."""


_RETRYABLE = (LLMRateLimitError, LLMServiceError, httpx.TransportError, httpx.TimeoutException)


_SYSTEM_PROMPT = """\
You are an expert resume data extraction engine. Your sole task is to extract \
structured information from resume text and return ONLY valid JSON — no markdown, \
no code fences, no explanation outside the JSON object.

Extraction rules:
1. Extract ONLY information explicitly present in the text; never infer, guess, or fabricate.
2. Use null for any scalar field not present in the resume.
3. Use [] for any list field with no matching content.
4. Preserve the candidate's original wording for summaries and responsibility bullet points.
5. Phone: preserve the original format exactly as written.
6. Duration: preserve original date text (e.g., "Jan 2020 – Mar 2022", "2019 – present").
7. Links: include all URLs found (LinkedIn, GitHub, portfolio, personal site, etc.).
8. Technical skills: programming languages, frameworks, libraries, tools, databases, platforms, cloud services.
9. Soft skills: interpersonal or professional traits (e.g., leadership, communication, problem-solving).
10. Languages: spoken/written human languages only — not programming languages \
(e.g., "English (native)", "Spanish (B2)").
11. Projects: personal, academic, or open-source projects distinct from employment history.
12. Awards: academic honors, scholarships, competition placements, professional recognition.\
"""

_SCHEMA_DESCRIPTION = json.dumps(
    {
        "contact": {
            "name": "string | null — Full name as written",
            "email": "string | null — Email address",
            "phone": "string | null — Phone number in original format",
            "location": "string | null — City, State/Country",
            "links": ["string — All profile/portfolio URLs found"],
        },
        "summary": "string | null — Verbatim professional summary or objective",
        "work_experience": [
            {
                "company": "string | null",
                "role": "string | null — Job title",
                "duration": "string | null — Date range as written",
                "location": "string | null — City or 'Remote'",
                "responsibilities": ["string — Each bullet point verbatim"],
            }
        ],
        "education": [
            {
                "degree": "string | null — Degree and field of study",
                "institution": "string | null",
                "year": "string | null — Graduation year or date range",
                "location": "string | null",
                "gpa": "string | null — GPA if mentioned",
            }
        ],
        "projects": [
            {
                "name": "string | null",
                "description": "string | null — What the project does",
                "technologies": ["string — Tech stack items"],
                "url": "string | null — Project URL if listed",
                "duration": "string | null — When built",
            }
        ],
        "technical_skills": ["string — Languages, frameworks, tools, databases, platforms"],
        "soft_skills": ["string — Interpersonal or professional traits"],
        "languages": ["string — Human languages (e.g., 'English (native)')"],
        "certifications": [
            {
                "name": "string | null",
                "issuer": "string | null — Issuing organization",
                "year": "string | null",
            }
        ],
        "awards": ["string — Recognitions, honors, scholarships, competition placements"],
        "raw_extraction_notes": ["string — Uncertainties or ambiguities you noticed"],
    },
    indent=2,
)



class BaseLLMClient(ABC):
    """Provider interface for resume extraction."""

    provider: str
    model: str

    @abstractmethod
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model response text."""


class OpenAIClient(BaseLLMClient):
    """OpenAI chat completions client."""

    provider = "openai"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required for the OpenAI provider.")

        import openai

        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            )
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(f"OpenAI rate limit: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LLMServiceError(f"OpenAI connection error: {exc}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMServiceError(f"OpenAI server error {exc.status_code}") from exc
            raise LLMExtractionError(
                f"OpenAI API error {exc.status_code}: {exc.message}"
            ) from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMExtractionError("OpenAI returned an empty response.")
        return content


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client using assistant-turn JSON prefill."""

    provider = "anthropic"

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is required for the Anthropic provider."
            )

        import anthropic

        self.model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        import anthropic

        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": "{"},
                ],
                timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            )
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(f"Anthropic rate limit: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMServiceError(f"Anthropic connection error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMServiceError(f"Anthropic server error {exc.status_code}") from exc
            raise LLMExtractionError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            ) from exc

        if message.stop_reason == "max_tokens":
            raise LLMExtractionError(
                "Anthropic response was truncated (max_tokens reached). "
                "The resume may be too long."
            )

        content = message.content[0].text if message.content else ""
        if not content:
            raise LLMExtractionError("Anthropic returned an empty response.")
        return "{" + content


class OllamaClient(BaseLLMClient):
    """Ollama local model client using the HTTP API."""

    provider = "ollama"

    def __init__(self) -> None:
        self.model = os.getenv("OLLAMA_MODEL", "mistral")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 429 or code >= 500:
                raise LLMServiceError(f"Ollama error {code}") from exc
            raise LLMExtractionError(f"Ollama error {code}") from exc

        content = response.json().get("message", {}).get("content")
        if not content:
            raise LLMExtractionError("Ollama returned an empty response.")
        return content




class LLMService:
    """Extract structured resume data with a configured LLM."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        self.client = client or _build_client()

    def extract_resume(self, resume_text: str) -> ResumeData:
        """Extract validated resume data from raw resume text."""
        user_prompt = (
            "Extract the resume into this exact JSON schema. "
            "All fields are required in the output.\n\n"
            f"Schema:\n{_SCHEMA_DESCRIPTION}\n\n"
            f"Resume text:\n{resume_text[:24_000]}"
        )
        response_text = self._retry_complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        try:
            payload = _parse_json(response_text)
            return ResumeData.model_validate(payload)
        except (ValidationError, TypeError) as exc:
            raise LLMExtractionError("LLM returned malformed resume data.") from exc

    def _retry_complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Call complete() with exponential backoff on transient errors."""
        max_attempts = max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return self.client.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt >= max_attempts - 1:
                    break
                wait = min(2.0 ** attempt + random.uniform(0, 0.5), 30.0)
                logger.bind(
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    wait_seconds=round(wait, 2),
                    error=str(exc),
                ).warning("llm_transient_error_retrying")
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]



def _build_client() -> BaseLLMClient:
    provider = os.getenv("RESUME_LLM_PROVIDER", "openai").lower().strip()
    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "ollama":
        return OllamaClient()
    raise LLMConfigurationError(
        f"Unknown RESUME_LLM_PROVIDER '{provider}'. "
        "Valid options: openai, anthropic, ollama."
    )


def _parse_json(response_text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    text = response_text.strip()

    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise LLMExtractionError("No JSON object found in LLM response.")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMExtractionError(f"Malformed JSON in LLM response: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMExtractionError("LLM response JSON root must be an object.")
    return parsed
