from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self):
        print("Chargement du modèle d'embeddings...")

        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        print("Modèle chargé avec succès.")

    def encode(self, texts):
        """
        Transforme une liste de textes en vecteurs.
        """
        return self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )


if __name__ == "__main__":

    service = EmbeddingService()

    texts = [
        "ما هو المفعول به؟",
        "Le Maroc est un pays d'Afrique."
    ]

    embeddings = service.encode(texts)

    print("\n===== TEST EMBEDDINGS =====")
    print("Nombre de textes :", len(texts))
    print("Dimension :", embeddings.shape)
    print("Premier vecteur :")
    print(embeddings[0])