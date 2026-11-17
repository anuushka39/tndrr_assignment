import config
from retrieval import retrieve
def normalize(text):
    """Lowercase and collapse all whitespace so PDF line breaks don't matter."""
    return " ".join(text.lower().split())


def is_hit(chunk, gold_passage):
    """A retrieved chunk is correct if it contains the gold passage text.
    Because we compare text (not chunk IDs), this works for any chunk size."""
    return normalize(gold_passage) in normalize(chunk["text"])


def validate_questions(pages, questions):
    """Warn if a gold passage can't be found in the document text.
    Without this, a typo would silently count as a retrieval failure."""
    all_text = normalize(" ".join(text for _, text in pages))
    for q in questions:
        if normalize(q["gold_passage"]) not in all_text:
            print(f"WARNING: gold passage not found in document: {q['question']}")


def evaluate(questions, index, chunks, k=config.TOP_K):
    """Return (recall@k, list of questions whose gold passage was NOT in the top k)."""
    hits = 0
    misses = []
    for q in questions:
        retrieved = retrieve(q["question"], index, chunks, k)
        if any(is_hit(c, q["gold_passage"]) for c in retrieved):
            hits += 1
        else:
            misses.append((q, retrieved))
    return hits / len(questions), misses