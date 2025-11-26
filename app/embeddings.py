import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from .db import SessionLocal
from .models import Resume

embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # fast & small[web:37]

faiss_index = None
id_map = []

def build_index():
    global faiss_index, id_map
    db: Session = SessionLocal()
    resumes = db.query(Resume).all()
    if not resumes:
        db.close()
        return

    texts = [r.raw_text for r in resumes]
    id_map = [r.id for r in resumes]

    embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    for r in resumes:
        r.embedding_dim = dim
    db.commit()
    db.close()

def search_similar(query_text: str, top_k: int = 5):
    if faiss_index is None:
        build_index()
    if faiss_index is None:
        return []

    q_vec = embed_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = faiss_index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        resume_id = id_map[idx]
        results.append((resume_id, float(score)))
    return results
