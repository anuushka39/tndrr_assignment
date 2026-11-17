import config

def load_pages(pdf_path):
    """Return a list of (page_number, text) for every page of the PDF."""
    reader = config.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append((i + 1, page.extract_text() or ""))
    return pages