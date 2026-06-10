"""Minimal FlashParse-compatible service for Veda's document pipeline.

Veda's gateway posts a multipart `file` to `{endpoint}/api/v1/pdf/process` and
expects a ZIP archive back containing a markdown file (`.md`) plus any images.
This satisfies that contract: PDFs are text-extracted with pypdf, text/markdown
files pass through, and the markdown is returned zipped. (No GPU/OCR stack —
enough to demonstrate parse → chunk → embed → insert end-to-end on text PDFs/docs.)
"""
import io
import zipfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

app = FastAPI()


def pdf_to_markdown(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    out = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(f"## Page {i}\n\n{text}")
    return "\n\n".join(out) or "# (no extractable text)"


@app.get("/health")
def health():
    return {"status": "ok", "service": "flashparse-lite"}


@app.post("/api/v1/pdf/process")
async def process(
    file: UploadFile = File(...),
    skip_captions: str = Form(default=None),
    dpi: str = Form(default=None),
):
    data = await file.read()
    name = (file.filename or "doc").lower()
    if name.endswith(".pdf"):
        try:
            md = pdf_to_markdown(data)
        except Exception as e:
            md = f"# (pdf parse error: {e})"
    else:
        md = data.decode("utf-8", errors="replace")
    if not md.strip():
        md = "# (empty document)"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("document.md", md)
    return Response(content=buf.getvalue(), media_type="application/zip")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8200)
