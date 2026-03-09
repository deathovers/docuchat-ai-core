from fastapi import FastAPI
from app.api.endpoints import chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DocuChat AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
