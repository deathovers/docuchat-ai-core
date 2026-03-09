from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.llm_service import LLMService
import os

app = FastAPI(title="DocuChat AI API")

# Initialize LLM Service (API Key should be in environment)
llm_service = LLMService(api_key=os.getenv("OPENAI_API_KEY", "dummy-key"))

class ChatRequest(BaseModel):
    query: str
    history: List[dict]
    session_id: str

@app.post("/v1/chat")
async def chat(request: ChatRequest):
    # In a real implementation, context_chunks would come from vector_service
    # For core logic demonstration, we use an empty list or mock data
    mock_context = [
        {
            "text": "The interest rate for the loan is 5.5% per annum.",
            "metadata": {"document_name": "Loan_Agreement.pdf", "page_number": 4}
        }
    ]
    
    response = await llm_service.get_chat_response(
        query=request.query,
        history=request.history,
        context_chunks=mock_context
    )
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
