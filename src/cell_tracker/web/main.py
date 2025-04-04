import os
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, HTTPException, status

BASE_DIRECTORY = Path("./shared_files").resolve()

app = FastAPI()

@app.get("/files/", response_model=List[Dict[str, str]])
@app.get("/files/{subpath:path}", response_model=List[Dict[str, str]])
async def list_files(subpath: str = "") -> List[Dict[str, str]]:
    """
    Lists files and directories within the secure BASE_DIRECTORY.
    The :path converter allows subpath to contain slashes '/'.
    """
    print(f"Request received for subpath: '{subpath}'")

    requested_path = (BASE_DIRECTORY / subpath).resolve()
    print(f"Resolved to path: {requested_path}")

    if BASE_DIRECTORY not in requested_path.parents and requested_path != BASE_DIRECTORY:
        print(f"Security alert: Path traversal attempt blocked: {requested_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found or access denied."
        )

    if not requested_path.is_dir():
        print(f"Path not found or not a directory: {requested_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Directory not found."
        )

    results = []
    try:
        for item_name in os.listdir(requested_path):
            item_path = requested_path / item_name
            item_type = "unknown"
            if item_path.is_file():
                item_type = "file"
            elif item_path.is_dir():
                item_type = "directory"

            results.append({"name": item_name, "type": item_type})

    except PermissionError:
        print(f"Permission denied for path: {requested_path}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied."
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred."
        )

    print(f"Returning {len(results)} items.")
    return results

@app.get("/")
async def read_root():
    return {"message": "Welcome! Try accessing /files/ to browse."}

if __name__ == "__main__":
    import uvicorn
    if not BASE_DIRECTORY.exists():
        print(f"Creating base directory: {BASE_DIRECTORY}")
        BASE_DIRECTORY.mkdir()
    elif not BASE_DIRECTORY.is_dir():
         print(f"Error: {BASE_DIRECTORY} exists but is not a directory!")
         exit(1)

    print(f"Starting server, serving files from: {BASE_DIRECTORY}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
