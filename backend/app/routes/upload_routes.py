"""
Document upload routes
"""
from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing"""
    # TODO: Implement document upload
    pass
