from fastapi import APIRouter, Body, Header
from fastapi.responses import StreamingResponse
from app.services.llm_service import llm_service
import json

router = APIRouter()

@router.post("/chat")
async def chat(
    query: str = Body(..., embed=True),
    session_id: str = Header(...)
):
    async def event_generator():
        async for chunk in llm_service.get_chat_response(query, session_id, []):
            if chunk.startswith("||SOURCES||"):
                sources = chunk.replace("||SOURCES||", "")
                yield f"data: {json.dumps({'type': 'sources', 'content': json.loads(sources)})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
