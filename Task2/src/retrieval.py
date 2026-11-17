import config
from embedding import embed
from google import genai  # reads GEMINI_API_KEY from the environment

def retrieve(query, index, chunks, k=config.TOP_K):
    """Return the k chunks most similar to the query, best first."""
    _, ids = index.search(embed([query]), k)
    return [chunks[i] for i in ids[0]]

def answer(question, retrieved):
    """Put the retrieved chunks in a prompt and ask the LLM."""
    context = "\n\n".join(f"[page {c['page']}] {c['text']}" for c in retrieved)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know. "
        "Mention the page number you used.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    client = genai.Client()
    return client.models.generate_content(model=config.LLM_MODEL, contents=prompt).text