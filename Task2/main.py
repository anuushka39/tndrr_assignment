"""
RAG from scratch: embeddings + FAISS, evaluated with recall@5 at two chunk sizes.
No LangChain / LlamaIndex.

Install:  pip install pypdf sentence-transformers faiss-cpu numpy google-genai
Run:      python rag.py document.pdf questions.json
Optional: python rag.py document.pdf questions.json "Ask a question here"
          (needs GEMINI_API_KEY set, only used for the final LLM answer)
"""
import json
import sys
import faiss
import numpy as np
from pypdf import PdfReader
import config
from sentence_transformers import SentenceTransformer
from src.loader import load_pages
from src.text_preprocessing import chunk_pages
from src.embedding import embed, build_index
from src.retrieval import retrieve, answer
from src.evaluate import normalize, is_hit, validate_questions, evaluate

CHUNK_SIZES = [75, 150]  # chunk length in WORDS. Keep both under ~200 words:
                         # this embedding model silently cuts text after 256 tokens.
OVERLAP = 20             # words shared between neighbouring chunks
TOP_K = 5                # recall@5
# EMBED_MODEL = "all-MiniLM-L6-v2"
# LLM_MODEL = "gemini-2.5-flash"  # change if this model name is retired

model = SentenceTransformer(config.EMBED_MODEL)

pdf_path, questions_path = sys.argv[1], sys.argv[2]
#1 Load-----------------
pages = load_pages(pdf_path)
with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
print(f"Loaded {len(pages)} pages and {len(questions)} questions")
validate_questions(pages, questions)

#2 Chunk------------
setups = {}  # chunk_size -> (chunks, index, recall)
for size in CHUNK_SIZES:
    chunks = chunk_pages(pages, size, OVERLAP)

#3 Vector storage ( embeed + storage)
    index = build_index(chunks)

#4 Retrieve + evaluation
    recall, misses = evaluate(questions, index, chunks)
    setups[size] = (chunks, index, recall)

    print(f"\n=== chunk size {size} words: {len(chunks)} chunks, recall@{TOP_K} = {recall:.2f} ===")
    for q, retrieved in misses:  # use these to pick your 2 failure cases
        print(f"\nMISSED: {q['question']}")
        print(f"  gold: {q['gold_passage'][:150]}")
        for rank, c in enumerate(retrieved[:2], start=1):
            print(f"  retrieved #{rank} (page {c['page']}): {c['text'][:150]}")

best_size = max(setups, key=lambda s: setups[s][2])
print(f"\nBest chunk size: {best_size} words")

if len(sys.argv) > 3:  # optional: answer one question with the LLM
    chunks, index, _ = setups[best_size]
    retrieved = retrieve(sys.argv[3], index, chunks)
    print("\nAnswer:", answer(sys.argv[3], retrieved))