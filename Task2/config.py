import json
import sys

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZES = [75, 150]  # chunk length in WORDS. Keep both under ~200 words:
                         # this embedding model silently cuts text after 256 tokens.
OVERLAP = 20             # words shared between neighbouring chunks
TOP_K = 5                # recall@5
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"  # change if this model name is retired

model = SentenceTransformer(EMBED_MODEL)