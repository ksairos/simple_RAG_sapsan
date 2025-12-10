import shutil
import uuid
import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, HTTPException
from starlette.background import BackgroundTasks

from app.rag import create_vector_store, generate_answer
from app.schemas import JobStatus, AnswerModel, QuestionModel

FILES_DIR = Path("files")
os.makedirs(FILES_DIR, exist_ok=True)

QDRANT_URL = os.environ.get("QDRANT_URL")
os.environ.get("OPENAI_API_KEY")

JOBS: Dict[str, dict] = {}

app = FastAPI()


@app.get("/")
async def health_check():
    return {"status": None}


@app.post("/upload_file")
async def upload_file(file: UploadFile):
    file_id = str(uuid.uuid4())
    file_path = FILES_DIR / file_id

    with file_path.open("wb") as dest_file:
        shutil.copyfileobj(file.file, dest_file)

    try:
        create_vector_store(file_id, str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

    return {"file_id": file_id, "filename": file.filename}


@app.post("/upload_question")
async def upload_question(question: QuestionModel, background_tasks: BackgroundTasks):
    question_id = str(uuid.uuid4())
    JOBS[question_id] = {"answer": None, "status": JobStatus.processing}

    background_tasks.add_task(
        process_question, question_id, question.file_id, question.question
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
        answer = generate_answer(file_id, question)

        JOBS[question_id]["status"] = JobStatus.success
        JOBS[question_id]["answer"] = answer

    except Exception as e:
        print(f"Error processing question {question}: {e}")
        JOBS[question_id]["status"] = JobStatus.error
