"""Pydantic request and response models for the resume parser API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ContactInfo(BaseModel):
    """Candidate contact details extracted from a resume."""

    name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    location: str | None = None
    links: list[str] = Field(default_factory=list)


class WorkExperience(BaseModel):
    """A single work experience entry."""

    company: str | None = None
    role: str | None = None
    duration: str | None = None
    location: str | None = None
    responsibilities: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """A single education entry."""

    degree: str | None = None
    institution: str | None = None
    year: str | None = None
    location: str | None = None
    gpa: str | None = None


class Certification(BaseModel):
    """A single certification entry."""

    name: str | None = None
    issuer: str | None = None
    year: str | None = None


class Project(BaseModel):
    """A single project entry."""

    name: str | None = None
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    url: str | None = None
    duration: str | None = None


class ResumeData(BaseModel):
    """Structured information extracted from a resume."""

    model_config = ConfigDict(extra="ignore")

    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str | None = None
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)
    raw_extraction_notes: list[str] = Field(default_factory=list)


class ResumeDocument(BaseModel):
    """Persisted resume extraction record."""

    document_id: str
    filename: str
    content_type: str | None = None
    uploaded_at: datetime
    extracted_data: ResumeData


class UploadResponse(ResumeDocument):
    """Response returned after a successful resume upload."""


class ResumeListItem(BaseModel):
    """Lightweight summary of a resume record for list endpoints."""

    document_id: str
    filename: str
    uploaded_at: datetime
    candidate_name: str | None = None


class ResumeListResponse(BaseModel):
    """Paginated list of resume summaries."""

    items: list[ResumeListItem]
    total: int
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    """Consistent API error response."""

    detail: str


class LLMRawResult(BaseModel):
    """Provider response before it is normalized into ResumeData."""

    provider: str
    model: str
    payload: dict[str, Any]
