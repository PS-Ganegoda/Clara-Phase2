from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.api.chat import router as chat_router

app = FastAPI(title="Chatbot API")

# 1️⃣ Add CORS middleware
origins = [
    "https://piumisaranga.vercel.app",  # your frontend URL
    # "http://localhost:3000",          # optional for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # allow only these domains
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, etc.
    allow_headers=["*"],        # all headers
)


app.include_router(chat_router, prefix="/chat", tags=["chat"])


@app.get("/")
def health():
    return {"status": "API running"}
