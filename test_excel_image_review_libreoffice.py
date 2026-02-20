#!/usr/bin/env python3
"""Unit tests for LibreOffice sheet-to-image conversion behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


class DummyOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class LibreOfficeConversionTests(unittest.TestCase):
    def setUp(self):
        self.work_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.work_dir.cleanup)
        self.excel_path = Path(self.work_dir.name) / "book.xlsx"
        self.excel_path.touch()

    def _build_reviewer(self):
        from excel_image_review import ExcelImageReviewer

        return ExcelImageReviewer(str(self.excel_path), output_dir=self.work_dir.name)

    def _mock_soffice_success(self, cmd, capture_output=True, timeout=120):
        out_dir = Path(cmd[cmd.index("--outdir") + 1])
        pdf_path = out_dir / f"{self.excel_path.stem}.pdf"
        pdf_path.touch()
        return type("Result", (), {"returncode": 0, "stderr": b""})()

    @patch("excel_image_review.OpenAI", new=DummyOpenAIClient)
    def test_libreoffice_uses_single_page_sheets_export(self):
        reviewer = self._build_reviewer()
        calls = []

        def fake_run(cmd, capture_output=True, timeout=120):
            calls.append(cmd)
            return self._mock_soffice_success(cmd, capture_output=capture_output, timeout=timeout)

        fake_page = object()
        with patch("subprocess.run", side_effect=fake_run), patch(
            "pdf2image.convert_from_path", return_value=[fake_page]
        ):
            reviewer._excel_to_images_libreoffice(["Sheet1"])

        self.assertTrue(calls, "Expected soffice conversion command to be executed")
        convert_arg = calls[0][3]
        self.assertIn(
            "SinglePageSheets",
            convert_arg,
            "LibreOffice export must enable SinglePageSheets to avoid partial sheet captures",
        )

    @patch("excel_image_review.OpenAI", new=DummyOpenAIClient)
    def test_raises_when_pdf_pages_do_not_match_sheet_count(self):
        reviewer = self._build_reviewer()

        with patch("subprocess.run", side_effect=self._mock_soffice_success), patch(
            "pdf2image.convert_from_path", return_value=[object(), object()]
        ):
            with self.assertRaises(RuntimeError):
                reviewer._excel_to_images_libreoffice(["Sheet1"])

    @patch("excel_image_review.OpenAI", new=DummyOpenAIClient)
    def test_process_with_limit_uses_full_sheet_list_for_libreoffice_mapping(self):
        reviewer = self._build_reviewer()
        all_sheets = ["S1", "S2", "S3", "S4"]
        captured = {"arg": None}
        fake_img = Image.new("RGB", (4, 4), "white")

        class DummyExcelFile:
            def __init__(self, path):
                self.sheet_names = all_sheets

        def fake_convert(names):
            captured["arg"] = list(names)
            return {name: fake_img for name in all_sheets}

        with patch("excel_image_review.pd.ExcelFile", DummyExcelFile), patch(
            "excel_image_review.shutil.which", return_value="/opt/homebrew/bin/soffice"
        ), patch.object(
            reviewer, "_excel_to_images_libreoffice", side_effect=fake_convert
        ), patch.object(
            reviewer, "review_image", return_value="ok"
        ), patch.object(
            reviewer, "excel_to_image", side_effect=AssertionError("should not fallback")
        ):
            reviewer.process_excel(limit=2)

        self.assertEqual(
            captured["arg"],
            all_sheets,
            "LibreOffice mapping should use full workbook sheet order even when limit is set",
        )
        self.assertEqual(list(reviewer.sheet_reviews.keys()), ["S1", "S2"])


if __name__ == "__main__":
    unittest.main()
