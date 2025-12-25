from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chatbot_services import get_bot_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/")
def chat(request: ChatRequest):
    reply = get_bot_response(request.message)
    return {"reply": reply}
