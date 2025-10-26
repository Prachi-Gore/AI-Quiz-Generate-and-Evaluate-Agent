from pydantic import BaseModel
from typing import List,Dict,Any

class BookSchema(BaseModel):
    id: str
    title: str
    author: str

class QuizRequest(BaseModel):
    book_id: str

class EvaluateRequest(BaseModel):
    book_id: str
    user_answers: List[str]
    quiz_details: List[Dict[str, Any]] 
