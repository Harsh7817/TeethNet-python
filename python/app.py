from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from uuid import uuid4
from pathlib import Path
import shutil
import os
import json
from dotenv import load_dotenv

# Import the processing functions directly
from tasks import process_image_direct, get_status, set_status

load_dotenv()

# Directories
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/data/uploads"))
RESULT_DIR = Path(os.environ.get("RESULT_DIR", "/data/results"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Depth->STL processing service (API)")

@app.post("/upload/")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    job_id = str(uuid4())
    safe_name = Path(file.filename).name
    fname = f"{job_id}_{safe_name}"
    save_path = UPLOAD_DIR / fname

    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    set_status(job_id, "QUEUED", "Job received and queued")
    
    # Run the processing in the background instead of using Celery
    background_tasks.add_task(process_image_direct, str(save_path), str(RESULT_DIR), job_id)

    return {"job_id": job_id, "celery_id": "background_task"}

@app.get("/status/{job_id}")
def status(job_id: str):
    info = get_status(job_id)
    if not info:
        return JSONResponse({"state": "UNKNOWN", "detail": "No such job_id"}, status_code=404)
    return JSONResponse(info)

@app.get("/download/{job_id}")
def download(job_id: str):
    info = get_status(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="No such job")
    if info.get("state") != "SUCCESS":
        raise HTTPException(status_code=404, detail="Result not ready")
    stl_path = info.get("result")
    if not stl_path or not Path(stl_path).exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    return FileResponse(path=stl_path, filename=Path(stl_path).name, media_type="application/sla")