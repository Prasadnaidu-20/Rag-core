from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
def embedder(document):

    doc_embeddings = model.encode(document)

    return doc_embeddings






