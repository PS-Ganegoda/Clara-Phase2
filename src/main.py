from fastapi import FastAPI
from app.api.chat import router as chat_router

app = FastAPI(title="Chatbot API")

app.include_router(chat_router, prefix="/chat", tags=["chat"])

@app.get("/")
def health():
    return {"status": "API running"}
