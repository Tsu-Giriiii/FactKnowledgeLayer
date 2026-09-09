import pdfplumber


def extract_pages(filepath: str) -> list[str]:
    """Return a list of per-page plain text strings (1 entry per PDF page)."""
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def chunk_pages(pages: list[str], pages_per_chunk: int):
    """Yield (start_page_number, end_page_number, combined_text) windows.

    Page numbers are 1-indexed to match what a human reading the PDF sees.
    Consecutive windows overlap by one page so facts that straddle a page
    boundary aren't lost.
    """
    n = len(pages)
    if n == 0:
        return
    step = max(pages_per_chunk - 1, 1)
    start = 0
    while start < n:
        end = min(start + pages_per_chunk, n)
        combined = "\n\n".join(
            f"[PAGE {i + 1}]\n{pages[i]}" for i in range(start, end)
        )
        yield start + 1, end, combined
        if end >= n:
            break
        start += step
