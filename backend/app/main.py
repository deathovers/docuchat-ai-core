from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import upload, chat

app = FastAPI(title="DocuChat AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/v1", tags=["upload"])
app.include_router(chat.router, prefix="/v1", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "DocuChat AI API is running"}
