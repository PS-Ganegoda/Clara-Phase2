from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chatbot_services import get_bot_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    reply = get_bot_response(request.message)
    return ChatResponse(reply=reply)
