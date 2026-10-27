"""Extract page provenance; flag empty extraction for review, without OCR."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect_pdf(path):
    from pypdf import PdfReader
    path=Path(path)
    version=hashlib.sha256(path.read_bytes()).hexdigest()
    reader=PdfReader(path)
    pages=[]
    for number,page in enumerate(reader.pages,1):
        text=page.extract_text() or ''
        pages.append({'document_id':path.stem,'content_sha256':version,
                      'page':number,'text':text,
                      'status':'text_extracted_review_layout' if text.strip() else 'needs_ocr_or_blank_review'})
    return pages


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('pdf',nargs='?',default=str(Path(__file__).parent/'fixtures/document-probe.pdf'))
    args=parser.parse_args()
    print(json.dumps(inspect_pdf(args.pdf),ensure_ascii=False,indent=2))
