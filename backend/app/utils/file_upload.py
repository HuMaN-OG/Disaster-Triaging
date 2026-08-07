import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException, status
from app.core.config import settings


async def save_uploaded_image(file: UploadFile) -> str:
    """
    Validate and save uploaded image file.
    Returns relative URL path for accessing the saved file.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded",
        )

    # Validate file extension
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}",
        )

    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image",
        )

    # Read content & check size
    contents = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    # Generate unique filename
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = settings.UPLOAD_DIR / unique_filename

    # Save to disk
    with open(file_path, "wb") as f:
        f.write(contents)

    # Reset file seek position
    await file.seek(0)

    return f"/uploads/{unique_filename}"
