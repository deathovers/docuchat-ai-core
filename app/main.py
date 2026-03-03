from fastapi import FastAPI
from app.api.endpoints import router as api_router
import uvicorn

app = FastAPI(title="DocuChat AI API")

app.include_router(api_router, prefix="/v1")

@app.get("/")
async def root():
    return {"message": "DocuChat AI API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
