from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from find_job_titles import FinderAcora
from pdfminder.high_level import extract_text

UPLOAD_DIR = Path() / 'uploads'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/uploadfile/")
async def create_upload_file(file_upload: UploadFile):

    data = await file_upload.read()
    save_to = UPLOAD_DIR / file_upload.filename
    with open(save_to, 'wb') as f:
        f.write(data)
    


    return {"filename":file_upload.filename} 