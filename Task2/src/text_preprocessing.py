def chunk_pages(pages, chunk_size, overlap):
    """Cut each page into overlapping windows of `chunk_size` words.
    Chunks never cross a page boundary, so each chunk has one page number."""
    chunks = []
    step = chunk_size - overlap
    for page_number, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            window = words[start:start + chunk_size]
            chunks.append({"text": " ".join(window), "page": page_number})
            if start + chunk_size >= len(words):  # reached the end of the page
                break
    return chunks

