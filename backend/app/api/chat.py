from fastapi import APIRouter

from app.models.chat import ChatRequest, ChatResponse
from app.services.rag_services import RAGService


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


# Chargement UNE SEULE FOIS
rag_service = RAGService()


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = rag_service.generate_response(
        request.message
    )

    return ChatResponse(
        response=result["response"],
        sources=result["sources"]
    )