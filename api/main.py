from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import upload, chat

app = FastAPI(title="DocuChat AI API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(upload.router, prefix="/v1", tags=["Upload"])
app.include_router(chat.router, prefix="/v1", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to DocuChat AI API"}
