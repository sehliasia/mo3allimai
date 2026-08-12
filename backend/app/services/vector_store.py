from pathlib import Path
import pickle

import faiss
import numpy as np

from app.services.document_loader import load_all_documents
from app.services.chunker import create_chunks
from app.services.embedding_service import EmbeddingService


VECTOR_DIR = Path("data/vector_store")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = VECTOR_DIR / "faiss.index"
CHUNKS_PATH = VECTOR_DIR / "chunks.pkl"


class VectorStore:

    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self):
        print("===== CONSTRUCTION DE L'INDEX FAISS =====")

        # 1. Charger les documents
        print("\nChargement des documents...")
        documents = load_all_documents()

        print(f"Nombre de documents/pages chargés : {len(documents)}")

        # 2. Créer les chunks
        print("\nCréation des chunks...")
        chunks = create_chunks(documents)

        print(f"Nombre de chunks : {len(chunks)}")

        if not chunks:
            raise ValueError("Aucun chunk trouvé.")

        # 3. Récupérer le texte des chunks
        texts = [chunk["texte"] for chunk in chunks]

        # 4. Générer les embeddings
        print("\nGénération des embeddings...")

        embedding_service = EmbeddingService()

        embeddings = embedding_service.encode(texts)

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        print(f"Dimension des embeddings : {embeddings.shape}")

        # 5. Créer l'index FAISS
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        # Normalisation pour la similarité cosinus
        faiss.normalize_L2(embeddings)

        index.add(embeddings)

        # 6. Sauvegarder l'index
        faiss.write_index(
            index,
            str(INDEX_PATH)
        )

        # 7. Sauvegarder les chunks
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)

        self.index = index
        self.chunks = chunks

        print("\n===== INDEXATION TERMINÉE =====")
        print(f"Vecteurs enregistrés : {index.ntotal}")
        print(f"Index : {INDEX_PATH}")
        print(f"Chunks : {CHUNKS_PATH}")


if __name__ == "__main__":
    store = VectorStore()
    store.build()