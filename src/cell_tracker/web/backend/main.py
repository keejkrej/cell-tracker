# main.py

from fastapi import FastAPI
from pathlib import Path

# Import the router object from the routers package/module
from .routers import files

app = FastAPI()

# Include the router from routers/files.py
# - prefix="/files": All routes in files.router will start with /files
# - tags=["files"]: Groups these routes under "files" in the OpenAPI docs (/docs)
app.include_router(files.router, prefix="/files", tags=["files"])

# You can keep other routes, like the root path, here
@app.get("/")
async def read_root():
    return {"message": "Welcome! Try accessing /files/ to browse."}

def main():
    import uvicorn

    # Optional: Check/create BASE_DIRECTORY on startup if running directly
    # Note: BASE_DIRECTORY is now defined in routers.files
    # If you need it here, you might import it or redefine it,
    # or move it to a central config module. For simplicity, let's
    # assume the uvicorn command is used primarily.
    base_dir_check = Path("./shared_files").resolve()
    if not base_dir_check.exists():
        print(f"[Main] Creating base directory: {base_dir_check}")
        base_dir_check.mkdir(parents=True, exist_ok=True) # Create if needed
    elif not base_dir_check.is_dir():
         print(f"[Main] Error: {base_dir_check} exists but is not a directory!")
         exit(1)

    print(f"[Main] Starting server...")
    # Run using the app instance defined in *this* file (main.py)
    uvicorn.run("cell_tracker.web.backend.main:app", host="127.0.0.1", port=8000, reload=True)

# --- Direct run (optional, uvicorn command is preferred) ---
if __name__ == "__main__":
    main()
    
# -------------------------------------------------------------