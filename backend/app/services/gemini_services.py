import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiService:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY n'est pas configurée dans le fichier .env"
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = "gemini-3.5-flash"

    def generate_response(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        if not response.text:
            return "Gemini n'a retourné aucune réponse."

        return response.text


if __name__ == "__main__":

    print("===== TEST GEMINI =====")

    service = GeminiService()

    response = service.generate_response(
        "Réponds en français : qu'est-ce que le RAG ?"
    )

    print("\n===== RÉPONSE GEMINI =====")
    print(response)