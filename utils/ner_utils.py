# utils/ner_utils.py
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("Downloading 'en_core_web_trf'...")
    download("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf")

EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EXCLUDED_ORGS = [
    "U.S. Securities and Exchange Commission",
    "SEC",
    "Nasdaq Stock Market",
    "New York Stock Exchange",
    "EDGAR",
    "Nasdaq",
    "Commission",
]
EXCLUDED_ORG_EMB = EMB_MODEL.encode(list(EXCLUDED_ORGS))

def extract_org_ner(text: str, similarity_threshold: float = 0.85):
    """
    Extracts distinct company/organization names from the text using spaCy NER,
    filtering out common financial/regulatory bodies and semantically similar duplicates.

    Args:
        text (str): Input text to extract entities from.
        similarity_threshold (float): Cosine similarity threshold to filter similar org names.

    Returns:
        List[Tuple[str, str]]: List of unique (entity_text, entity_label) tuples.
    """
    doc = nlp(text)
    unique_orgs = []
    unique_embeddings = []

    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue

        org_name = ent.text.strip()
        org_embedding = EMB_MODEL.encode(org_name)

        # Filter out excluded organizations
        if any(sim >= similarity_threshold for sim in cosine_similarity([org_embedding], EXCLUDED_ORG_EMB)[0]):
            continue

        # Filter out duplicates (semantically similar orgs)
        if unique_embeddings:
            if any(sim >= similarity_threshold for sim in cosine_similarity([org_embedding], unique_embeddings)[0]):
                continue

        unique_orgs.append((org_name, ent.label_))
        unique_embeddings.append(org_embedding)

    return unique_orgs
