from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from find_job_titles import FinderAcora
from pdfminer.high_level import extract_text
import uvicorn

UPLOAD_DIR = Path() / 'uploads'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
finder = FinderAcora()

@app.post("/uploadfile/")
async def create_upload_file(file_upload: UploadFile):
    global file_name
    data = await file_upload.read()
    save_to = UPLOAD_DIR / file_upload.filename
    with open(save_to, 'wb') as f:
        f.write(data)
    
    file_name = file_upload.filename

    return {"filename":file_upload.filename} 

@app.post("/pulljobtitles")
async def pull_job_titles():
    global job_list
    # extract information from the resume
    text = extract_text(rf'{UPLOAD_DIR}/{file_name}')
    
    # run the job title parser
    jobs = finder.findall(text)
    
    # pull job titles from extracted resume
    job_list = []
    for job_title in jobs:
        job_list.append(job_title[2])


    return {"job_list": job_list}

@app.post("/pullskills")
async def pull_skills():


    return
    
#if __name__ == '__main__':
#    uvicorn.run(app)