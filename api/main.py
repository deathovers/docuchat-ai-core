from fastapi import FastAPI
from api.routes import upload, chat
from api.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(upload.router, prefix="/v1", tags=["upload"])
app.include_router(chat.router, prefix="/v1", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "DocuChat AI API is running"}
