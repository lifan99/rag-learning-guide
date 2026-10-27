import unittest
from pathlib import Path
from document_probe import inspect_pdf


class DocumentProbeTests(unittest.TestCase):
    def test_page_identity_and_image_only_fallback(self):
        path=Path(__file__).parent/'fixtures/document-probe.pdf'
        rows=inspect_pdf(path)
        self.assertEqual([r['page'] for r in rows],[1,2])
        self.assertIn('Annual subsidy cap: 2000 yuan.',rows[0]['text'])
        self.assertIn('Manager approval is required before enrollment.',rows[0]['text'])
        self.assertEqual(rows[1]['status'],'needs_ocr_or_blank_review')
        self.assertEqual(rows[1]['text'].strip(),'')
        self.assertEqual(rows[0]['content_sha256'],rows[1]['content_sha256'])

    def test_repeat_preserves_provenance(self):
        path=Path(__file__).parent/'fixtures/document-probe.pdf'
        self.assertEqual(inspect_pdf(path),inspect_pdf(path))


if __name__ == '__main__':
    unittest.main()
