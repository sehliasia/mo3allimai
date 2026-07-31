from fastapi import APIRouter

from app.models.chat import ChatRequest, ChatResponse
from app.services.openai_service import OpenAIService


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):

    reply = OpenAIService.generate_response(
        request.message
    )

    return ChatResponse(
        response=reply
    )