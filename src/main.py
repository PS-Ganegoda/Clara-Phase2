from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.services.chatbot_services import get_model

app = FastAPI()

# Include chat routes
app.include_router(chat_router)

# Warm up model on startup
@app.on_event("startup")
def startup_event():
    get_model()
