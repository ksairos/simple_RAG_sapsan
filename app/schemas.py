from enum import Enum
from typing import Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    """
    Статус обработки вопроса
    """

    processing = "processing"
    success = "success"
    error = "error"


class AnswerModel(BaseModel):
    """
    Схема получения ответа
    """

    status: JobStatus
    answer: Optional[str]


class QuestionModel(BaseModel):
    """
    Схема загрузки вопроса
    """

    question: str
    file_id: str
