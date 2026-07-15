from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from smart_traffic_agent.rag.protocol_document_loader import build_protocol_document_chunks


class ProtocolDocumentLoaderTests(unittest.TestCase):
    def test_build_chunks_from_mixed_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "doc.html").write_text(
                "<html><head><title>Protocol Commands</title></head>"
                "<body><h1>Read Status</h1><p>Use function code to read machine status.</p></body></html>",
                encoding="utf-8",
            )
            (root / "notes.txt").write_text("Start device\nRead register\nStop device", encoding="utf-8")
            self._write_docx(root / "manual.docx", ["Operation mode", "Start, pause, resume"])

            output = root / "chunks.jsonl"
            chunks = build_protocol_document_chunks(
                output,
                protocol="demo",
                inputs=[root],
                max_chars=200,
                overlap=20,
            )

            self.assertEqual(len(chunks), 3)
            self.assertTrue(output.exists())
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["protocol"] for row in rows}, {"demo"})
            self.assertEqual({row["knowledge_type"] for row in rows}, {"document_source"})
            self.assertTrue(any(row["source_type"] == "html_document" for row in rows))
            self.assertTrue(any(row["source_type"] == "docx_document" for row in rows))
            self.assertTrue(any(row["source_type"] == "text_document" for row in rows))

    def _write_docx(self, path: Path, paragraphs: list[str]) -> None:
        body = "".join(
            f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
            for paragraph in paragraphs
        )
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}</w:body>"
            "</w:document>"
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", document_xml)


if __name__ == "__main__":
    unittest.main()
