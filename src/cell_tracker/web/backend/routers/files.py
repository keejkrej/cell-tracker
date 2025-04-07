# routers/files.py

import os
from pathlib import Path
from typing import List, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

# --- Configuration ---
BASE_DIRECTORY = Path("./shared_files").resolve()
# -------------------

router = APIRouter()

# --- Route for Browse (Listing Contents) ---

# Changed route prefix to /browse/
@router.get("/browse/", response_model=List[Dict[str, str]])
@router.get("/browse/{subpath:path}", response_model=List[Dict[str, str]])
async def browse_files(subpath: str = "") -> List[Dict[str, str]]: # Renamed function for clarity
    """
    Provides a listing representation of a directory's contents within BASE_DIRECTORY.
    Accessed via /files/browse/ or /files/browse/subdirectory
    """
    print(f"[Router] Request received for BROWSE subpath: '{subpath}'")
    requested_path = (BASE_DIRECTORY / subpath).resolve()
    print(f"[Router] Resolved path for Browse: {requested_path}")

    # *** SECURITY CHECK *** (Keep as before)
    if BASE_DIRECTORY not in requested_path.parents and requested_path != BASE_DIRECTORY:
        if not str(requested_path).startswith(str(BASE_DIRECTORY)):
            print(f"[Router] Security alert: Path traversal attempt blocked: {requested_path}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found or access denied.")

    # Check if the path exists and IS A DIRECTORY for Browse
    if not requested_path.is_dir():
        print(f"[Router] Path not found or not a directory for Browse: {requested_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Directory not found.")

    # List directory contents (Keep logic as before)
    results = []
    try:
        for item_name in os.listdir(requested_path):
            item_path = requested_path / item_name
            item_type = "unknown"
            # Optionally add HATEOAS links here in the future if desired
            if item_path.is_file():
                item_type = "file"
            elif item_path.is_dir():
                item_type = "directory"
            results.append({"name": item_name, "type": item_type})
    except PermissionError:
        print(f"[Router] Permission denied for path: {requested_path}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied.")
    except Exception as e:
        print(f"[Router] An error occurred during Browse: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")

    print(f"[Router] Returning {len(results)} items from Browse.")
    return results


# --- Route for Raw File Content ---

# Changed route prefix to /raw/
@router.get("/raw/{file_path:path}")
async def get_raw_file(file_path: str): # Renamed function for clarity
    """
    Serves the raw content representation of a file from BASE_DIRECTORY.
    Accessed via /files/raw/path/to/file.jpg
    """
    print(f"[Router] Request received for RAW content: '{file_path}'")
    requested_path = (BASE_DIRECTORY / file_path).resolve()
    print(f"[Router] Resolved path for raw content: {requested_path}")

    # *** SECURITY CHECK *** (Repeat the same check here)
    if BASE_DIRECTORY not in requested_path.parents and requested_path != BASE_DIRECTORY:
        if not str(requested_path).startswith(str(BASE_DIRECTORY)):
            print(f"[Router] Security alert: Path traversal attempt blocked for raw content: {requested_path}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found or access denied.")

    # *** FILE CHECK *** (Ensure it's a file)
    if not requested_path.is_file():
        print(f"[Router] Path is not a file or does not exist for raw content: {requested_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    # Return the file content using FileResponse
    try:
        # Add cache headers here if needed for better REST compliance
        return FileResponse(path=requested_path)
    except PermissionError:
         print(f"[Router] Permission denied for file: {requested_path}")
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied.")
    except Exception as e:
        print(f"[Router] Error serving raw file {requested_path}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not serve file.")