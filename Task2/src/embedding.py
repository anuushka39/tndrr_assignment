import config 

def embed(texts):
    """Turn a list of strings into a float32 matrix, one unit-length row per string."""
    vectors = config.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return config.np.array(vectors, dtype="float32")


def build_index(chunks):
    """Embed all chunks and put them in a FAISS index.
    Vectors are unit length, so inner product == cosine similarity."""
    vectors = embed([c["text"] for c in chunks])
    index = config.faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index