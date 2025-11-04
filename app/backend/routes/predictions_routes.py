# routes/prediction_routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from controller import classify_controller

router = APIRouter(prefix="/predict")

class TextInput(BaseModel):
    text: str

@router.post("/")
def predict_text(input: TextInput):
    result = classify_controller.classify_both(input.text)
    return result
