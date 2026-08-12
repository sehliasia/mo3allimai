from pathlib import Path
from pypdf import PdfReader


# Dossier contenant les documents PDF
DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"


def extract_pdf(pdf_path: Path) -> list[dict]:
    """
    Extrait le texte d'un PDF page par page.

    Chaque page devient un document avec :
    - texte
    - source
    - page
    - niveau
    """

    documents = []

    # Le niveau correspond au nom du dossier
    niveau = pdf_path.parent.name

    try:
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            text = text.strip()

            # On ignore les pages complètement vides
            if not text:
                continue

            documents.append(
                {
                    "texte": text,
                    "source": pdf_path.name,
                    "page": page_number,
                    "niveau": niveau,
                }
            )

    except Exception as e:
        print(f"Erreur avec {pdf_path.name}: {e}")

    return documents


def load_all_documents() -> list[dict]:
    """
    Parcourt tous les PDF du dossier documents.
    """

    all_documents = []

    pdf_files = list(DOCUMENTS_DIR.rglob("*.pdf"))

    print(f"PDF trouvés : {len(pdf_files)}")

    for pdf_path in pdf_files:

        print(f"Lecture : {pdf_path.name}")

        documents = extract_pdf(pdf_path)

        all_documents.extend(documents)

    print(f"Pages avec texte extraites : {len(all_documents)}")

    return all_documents


if __name__ == "__main__":

    documents = load_all_documents()

    print("\n===== TEST =====")

    if documents:

        first_document = documents[0]

        print("Source :", first_document["source"])
        print("Page   :", first_document["page"])
        print("Niveau :", first_document["niveau"])

        print("\nTexte :")
        print(first_document["texte"][:1000])

    else:
        print("Aucun texte extrait.")