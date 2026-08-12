from typing import List, Dict


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Découpe un texte en chunks avec chevauchement.
    """

    if not text:
        return []

    text = text.strip()

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - chunk_overlap

    return chunks


def create_chunks(documents: List[Dict]) -> List[Dict]:
    """
    Transforme les pages extraites en chunks.

    Les métadonnées de la page sont conservées.
    """

    chunks = []

    for document in documents:

        text = document["texte"]

        text_chunks = split_text(text)

        for index, chunk in enumerate(text_chunks):

            chunks.append(
                {
                    "texte": chunk,
                    "source": document["source"],
                    "page": document["page"],
                    "niveau": document["niveau"],
                    "chunk_id": index
                }
            )

    return chunks
if __name__ == "__main__":

    from document_loader import load_all_documents

    documents = load_all_documents()

    chunks = create_chunks(documents)

    print("\n====================")
    print("TEST CHUNKING")
    print("====================")

    print("Nombre de pages :", len(documents))
    print("Nombre de chunks :", len(chunks))

    if chunks:

        print("\nPremier chunk :")
        print(chunks[0]["texte"])

        print("\nMétadonnées :")
        print("Source :", chunks[0]["source"])
        print("Page :", chunks[0]["page"])
        print("Niveau :", chunks[0]["niveau"])
        print("Chunk ID :", chunks[0]["chunk_id"])