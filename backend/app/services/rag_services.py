from app.services.retriever import Retriever
from app.services.gemini_service import GeminiService


class RAGService:

    def __init__(self):

        print("Initialisation du RAG...")

        self.retriever = Retriever()
        self.gemini = GeminiService()

        print("RAG initialisé avec succès.")

    def generate_response(
        self,
        question: str,
        top_k: int = 5
    ):

        # ==========================================
        # 1. Recherche avec FAISS
        # ==========================================

        results = self.retriever.retrieve(
            question,
            top_k=top_k
        )

        if not results:

            return {
                "response": (
                    "Je n'ai pas trouvé d'information pertinente "
                    "dans les documents disponibles."
                ),
                "sources": []
            }

        # ==========================================
        # 2. Construire le contexte
        # ==========================================

        context_parts = []

        for i, result in enumerate(results, start=1):

            context_parts.append(
                f"""
--- DOCUMENT {i} ---

Source : {result['source']}
Page : {result['page']}
Niveau : {result['niveau']}

{result['texte']}
"""
            )

        context = "\n".join(context_parts)

        # ==========================================
        # 3. Prompt RAG
        # ==========================================

        prompt = f"""
Tu es Mo3allimai, un assistant pédagogique spécialisé
dans l'enseignement de la langue arabe.

Tu dois répondre à la question de l'utilisateur en utilisant
UNIQUEMENT les informations présentes dans le CONTEXTE.

RÈGLES :

1. Ne fabrique aucune information.
2. Si la réponse n'est pas présente dans le contexte,
   indique que l'information n'a pas été trouvée.
3. Réponds dans la même langue que la question.
4. Donne une réponse claire et pédagogique.
5. Ne mentionne pas que tu es un modèle d'IA.
6. Utilise les documents comme source principale.

QUESTION :

{question}

CONTEXTE :

{context}

RÉPONSE :
"""

        # ==========================================
        # 4. Gemini
        # ==========================================

        response = self.gemini.generate_response(prompt)

        # ==========================================
        # 5. Sources
        # ==========================================

        sources = []

        for result in results:

            sources.append({
                "document": result["source"],
                "page": result["page"],
                "score": round(
                    float(result.get("score", 0)),
                    4
                )
            })

        return {
            "response": response,
            "sources": sources
        }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print("===== TEST DU RAG =====")

    rag = RAGService()

    question = input("\nQuestion : ")

    result = rag.generate_response(question)

    print("\n===== RÉPONSE =====")

    print(result["response"])

    print("\n===== SOURCES =====")

    for source in result["sources"]:

        print(
            f"- {source['document']} "
            f"(page {source['page']}, "
            f"score={source['score']})"
        )