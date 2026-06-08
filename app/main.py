
from __future__ import annotations

import functools
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.models import ErrorResponse, ResumeListResponse, UploadResponse
from app.services.llm_service import LLMConfigurationError, LLMExtractionError, LLMService
from app.services.parser import EmptyResumeError, ResumeParser, UnsupportedFileTypeError
from app.services.storage import StorageService
from app.utils.logger import configure_logging, get_logger

load_dotenv()
configure_logging("api")

logger = get_logger(__name__)
storage_service = StorageService()
resume_parser = ResumeParser()


@functools.lru_cache(maxsize=1)
def _get_llm_service() -> LLMService:
    """Return the shared LLMService instance (created lazily on first request)."""
    return LLMService()



@asynccontextmanager
async def lifespan(app: FastAPI):  
    """Log startup state and perform a best-effort LLM config probe."""
    try:
        svc = _get_llm_service()
        logger.bind(provider=svc.client.provider, model=svc.client.model).info(
            "llm_service_ready"
        )
    except LLMConfigurationError as exc:

        logger.bind(reason=str(exc)).warning("llm_not_configured")

    logger.info("resume_parser_api_started")
    yield
    logger.info("resume_parser_api_stopped")


app = FastAPI(
    title="Resume Parser API",
    description=(
        "Upload a PDF or DOCX resume and extract structured data with an LLM.\n\n"
        "Supported providers: OpenAI, Anthropic Claude, Ollama (local)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return a minimal liveness response."""
    return {"status": "ok"}



@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and parse a resume",
    tags=["resumes"],
)
async def upload_resume(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a PDF/DOCX resume, extract structured fields via LLM, and persist results."""

    filename = file.filename or ""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    extension = Path(filename).suffix.lower()
    if extension not in ResumeParser.supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{extension}'. Only PDF and DOCX are accepted.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(content) > _max_upload_bytes():
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {os.getenv('MAX_UPLOAD_MB', '10')} MB upload limit.",
        )


    try:
        ResumeParser.validate_magic_bytes(content, extension)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    document_id, file_path = storage_service.save_upload(file, content)
    logger.bind(
        document_id=document_id,
        filename=filename,
        content_type=file.content_type,
        size_bytes=len(content),
    ).info("resume_upload_saved")

    try:
        resume_text = resume_parser.extract_text(file_path)
        extracted_data = _get_llm_service().extract_resume(resume_text)
        document = storage_service.save_resume(
            document_id=document_id,
            filename=filename,
            content_type=file.content_type,
            extracted_data=extracted_data,
        )
    except (UnsupportedFileTypeError, EmptyResumeError) as exc:
        _delete_upload(file_path, document_id)
        logger.bind(document_id=document_id, reason=str(exc)).warning(
            "resume_text_extraction_failed"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except LLMConfigurationError as exc:
        _delete_upload(file_path, document_id)
        logger.bind(document_id=document_id, reason=str(exc)).error(
            "llm_configuration_error"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMExtractionError as exc:
        _delete_upload(file_path, document_id)
        logger.bind(document_id=document_id, reason=str(exc)).opt(exception=True).error(
            "llm_extraction_failed"
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The LLM could not extract structured resume data.",
        ) from exc
    except Exception as exc:
        _delete_upload(file_path, document_id)
        logger.bind(document_id=document_id, reason=str(exc)).opt(exception=True).error(
            "resume_processing_failed"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resume processing failed.",
        ) from exc

    logger.bind(document_id=document_id).info("resume_extracted")
    return UploadResponse.model_validate(document.model_dump())


@app.get(
    "/api/resumes",
    response_model=ResumeListResponse,
    summary="List parsed resumes",
    tags=["resumes"],
)
def list_resumes(
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
) -> ResumeListResponse:
    """Return a paginated list of previously parsed resumes, newest first."""
    items = storage_service.list_resumes(limit=limit, offset=offset)
    total = storage_service.total_resumes()
    return ResumeListResponse(items=items, total=total, limit=limit, offset=offset)


@app.get(
    "/api/resume/{document_id}",
    response_model=UploadResponse,
    summary="Get parsed resume by document id",
    tags=["resumes"],
)
def get_resume(document_id: str) -> UploadResponse:
    """Retrieve a previously extracted resume by its unique document id."""
    document = storage_service.get_resume(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume '{document_id}' not found.",
        )
    return UploadResponse.model_validate(document.model_dump())


@app.delete(
    "/api/resume/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a parsed resume",
    tags=["resumes"],
)
def delete_resume(document_id: str) -> None:
    """Delete a stored resume record and its original uploaded file."""
    deleted = storage_service.delete_resume(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume '{document_id}' not found.",
        )



def _max_upload_bytes() -> int:
    try:
        return int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    except ValueError:
        return 10 * 1024 * 1024


def _delete_upload(file_path: Path, document_id: str) -> None:
    """Best-effort removal of an orphaned upload file after a processing failure."""
    try:
        file_path.unlink(missing_ok=True)
    except OSError:
        logger.bind(document_id=document_id, path=str(file_path)).warning(
            "orphaned_upload_cleanup_failed"
        )
