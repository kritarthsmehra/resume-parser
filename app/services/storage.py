"""Persistence helpers for uploaded files and extracted resume records."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.models import ResumeData, ResumeDocument, ResumeListItem
from app.utils.logger import get_logger

_logger = get_logger(__name__)


class StorageService:
    """Store uploaded files and parsed resume records on disk."""

    def __init__(self, base_dir: Path | str = "data") -> None:
        self.base_dir = Path(base_dir)
        self.upload_dir = self.base_dir / "uploads"
        self.resume_dir = self.base_dir / "resumes"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.resume_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, upload: UploadFile, content: bytes) -> tuple[str, Path]:
        """Persist the original file and return its document id and path."""
        document_id = str(uuid.uuid4())
        safe_name = _safe_filename(upload.filename or "resume")
        extension = Path(safe_name).suffix.lower()
        file_path = self.upload_dir / f"{document_id}{extension}"
        file_path.write_bytes(content)
        return document_id, file_path

    def save_resume(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str | None,
        extracted_data: ResumeData,
    ) -> ResumeDocument:
        """Persist extracted resume data as JSON using an atomic write."""
        document = ResumeDocument(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            uploaded_at=datetime.now(timezone.utc),
            extracted_data=extracted_data,
        )
        target = self._resume_path(document_id)
        temp = target.with_suffix(".tmp")
        temp.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(target)
        return document

    def get_resume(self, document_id: str) -> ResumeDocument | None:
        """Read a previously extracted resume by id."""
        if not _is_uuid(document_id):
            return None
        path = self._resume_path(document_id)
        if not path.exists():
            return None
        return ResumeDocument.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def delete_resume(self, document_id: str) -> bool:
        """Delete resume record and its original upload. Returns True if deleted."""
        if not _is_uuid(document_id):
            return False
        path = self._resume_path(document_id)
        if not path.exists():
            return False
        path.unlink()
        for upload_file in self.upload_dir.glob(f"{document_id}.*"):
            upload_file.unlink(missing_ok=True)
        return True

    def list_resumes(self, limit: int = 50, offset: int = 0) -> list[ResumeListItem]:
        """Return resume summaries sorted by upload time descending."""
        paths = sorted(
            self.resume_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        items: list[ResumeListItem] = []
        for path in paths[offset : offset + limit]:
            try:
                doc = ResumeDocument.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                items.append(
                    ResumeListItem(
                        document_id=doc.document_id,
                        filename=doc.filename,
                        uploaded_at=doc.uploaded_at,
                        candidate_name=doc.extracted_data.contact.name,
                    )
                )
            except Exception as exc:
                _logger.bind(path=str(path), reason=str(exc)).warning(
                    "resume_record_deserialization_failed"
                )
                continue
        return items

    def total_resumes(self) -> int:
        """Return the total count of stored resume records."""
        return sum(1 for _ in self.resume_dir.glob("*.json"))

    def _resume_path(self, document_id: str) -> Path:
        return self.resume_dir / f"{document_id}.json"


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or "resume"
