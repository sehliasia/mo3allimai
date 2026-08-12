from pathlib import Path
import pickle

import faiss
import numpy as np

from app.services.embedding_service import EmbeddingService


VECTOR_DIR = Path("data/vector_store")

INDEX_PATH = VECTOR_DIR / "faiss.index"
CHUNKS_PATH = VECTOR_DIR / "chunks.pkl"


class Retriever:

    def __init__(self):
        print("Chargement du retriever...")

        # Vérifier que les fichiers existent
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Index FAISS introuvable : {INDEX_PATH}"
            )

        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Chunks introuvables : {CHUNKS_PATH}"
            )

        # Charger FAISS
        self.index = faiss.read_index(str(INDEX_PATH))

        # Charger les chunks
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        # Service d'embeddings
        self.embedding_service = EmbeddingService()

        print("Retriever chargé avec succès.")
        print(f"Nombre de vecteurs : {self.index.ntotal}")
        print(f"Nombre de chunks : {len(self.chunks)}")

    def retrieve(self, query: str, top_k: int = 5):
        """
        Recherche les chunks les plus pertinents
        pour une question utilisateur.
        """

        if not query or not query.strip():
            return []

        # Limiter top_k au nombre de vecteurs disponibles
        top_k = min(top_k, self.index.ntotal)

        # 1. Transformer la question en embedding
        query_embedding = self.embedding_service.encode([query])

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # 2. Normaliser pour utiliser la similarité cosinus
        faiss.normalize_L2(query_embedding)

        # 3. Recherche dans FAISS
        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        # 4. Construire les résultats
        results = []

        for score, index in zip(scores[0], indices[0]):

            # FAISS peut retourner -1 si aucun résultat
            if index == -1:
                continue

            chunk = self.chunks[index]

            results.append({
                "score": float(score),
                "texte": chunk.get("texte", ""),
                "source": chunk.get("source", ""),
                "page": chunk.get("page", ""),
                "niveau": chunk.get("niveau", ""),
                "chunk_id": chunk.get("chunk_id", index)
            })

        return results


if __name__ == "__main__":

    print("\n===== TEST DU RETRIEVER =====\n")

    retriever = Retriever()

    query = input("Entrez votre question : ")

    results = retriever.retrieve(
        query,
        top_k=5
    )

    print("\n===== RÉSULTATS =====\n")

    if not results:
        print("Aucun résultat trouvé.")

    else:
        for i, result in enumerate(results, start=1):

            print(f"--- Résultat {i} ---")
            print(f"Score    : {result['score']:.4f}")
            print(f"Source   : {result['source']}")
            print(f"Page     : {result['page']}")
            print(f"Niveau   : {result['niveau']}")
            print(f"Chunk ID : {result['chunk_id']}")
            print("\nTexte :")
            print(result["texte"][:1000])
            print("\n")