import uuid
import os
from typing import Dict

from fastapi import FastAPI, UploadFile, HTTPException
from starlette.background import BackgroundTasks

from app.schemas import JobStatus, AnswerModel, QuestionModel

FILES_DIR = "files"
os.makedirs(FILES_DIR, exist_ok=True)

SAVED_FILE: Dict[str, str] = {}
JOBS: Dict[str, dict] = {}

app = FastAPI()

@app.get("/")
async def health_check():
    return {"status": None}

@app.post("/upload_file")
async def upload_file(file: UploadFile):

    file_id = str(uuid.uuid4())
    file_path = os.path.join(FILES_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    SAVED_FILE[file_id] = file_path

    return {"file_id": file_id}

@app.post("/upload_question")
async def upload_question(question: QuestionModel, background_tasks: BackgroundTasks):

    if not SAVED_FILE:
        raise HTTPException(status_code=400, detail="No files uploaded")

    question_id = str(uuid.uuid4())
    JOBS[question_id] = {
        "answer": None,
        "status": JobStatus.processing
    }
    print(JOBS)

    background_tasks.add_task(
        process_question,
        question_id,
        question.file_id,
        question.question
    )

    return {"question_id": question_id}

@app.get("/get_answer/{question_id}", response_model=AnswerModel)
async def get_answer(question_id: str):

    job = JOBS.get(question_id)

    if not job:
        raise HTTPException(status_code=404, detail="Question not found")

    return {"status": job["status"], "answer": job.get("answer")}


def process_question(question_id: str, file_id: str, question: str):
    try:
        file_path = SAVED_FILE.get(file_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")

        simulated_answer = (
            f"Временный ответ на: '{question}'. "
            f"Контекст файла ID: {file_id}"
        )

        JOBS[question_id]["status"] = JobStatus.success
        JOBS[question_id]["answer"] = simulated_answer

    except Exception as e:
        print(f"Error processing question {question}: {e}")
        JOBS[question_id]["status"] = JobStatus.error