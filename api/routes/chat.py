from fastapi import APIRouter, HTTPException
from api.models.schemas import ChatRequest, ChatResponse
from api.services.rag_engine import RAGEngine

router = APIRouter()
rag_engine = RAGEngine()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = await rag_engine.get_answer(request.session_id, request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
