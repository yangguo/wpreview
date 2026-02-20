#!/usr/bin/env python3
"""Tests for Chinese review prompt and DOCX report generation."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image
from docx import Document


class DummyOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ReportDocxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.excel_path = Path(self.tmp.name) / "book.xlsx"
        self.excel_path.touch()

    def _reviewer(self):
        from excel_image_review import ExcelImageReviewer

        return ExcelImageReviewer(str(self.excel_path), output_dir=self.tmp.name)

    def test_generate_report_outputs_docx_and_parses_markdown(self):
        reviewer = self._reviewer()

        image_path = Path(self.tmp.name) / "sheet.png"
        Image.new("RGB", (40, 40), "white").save(image_path)

        reviewer.sheet_images = {"Sheet1": image_path}
        reviewer.sheet_reviews = {
            "Sheet1": "## 二级标题\n- **问题A**：字段拼写不一致\n- 影响：导致误解\n- 修改为：统一术语\n\n`CODE-1`"
        }

        report_path = reviewer.generate_report()

        self.assertTrue(Path(report_path).exists())
        self.assertEqual(Path(report_path).suffix, ".docx")

        doc = Document(str(report_path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        self.assertIn("二级标题", text)
        self.assertIn("问题A", text)
        self.assertIn("CODE-1", text)
        self.assertNotIn("影响：", text)

    def test_review_prompt_requires_chinese_output(self):
        reviewer = self._reviewer()
        captured = {"prompt": ""}

        class FakeResponse:
            class Choice:
                class Message:
                    content = "ok"

                message = Message()

            choices = [Choice()]

        class FakeCompletions:
            def create(self, **kwargs):
                user_content = kwargs["messages"][1]["content"]
                captured["prompt"] = user_content[0]["text"]
                return FakeResponse()

        reviewer.client = type(
            "FakeClient",
            (),
            {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
        )()

        reviewer.review_image(Image.new("RGB", (30, 30), "white"), "Sheet1")
        self.assertIn("中文", captured["prompt"])
        self.assertIn("审计程序要求", captured["prompt"])
        self.assertIn("测试过程", captured["prompt"])
        self.assertIn("整改意见", captured["prompt"])
        self.assertIn("直接给出问题", captured["prompt"])
        self.assertIn("具体位置", captured["prompt"])
        self.assertIn("原文", captured["prompt"])
        self.assertIn("修改为", captured["prompt"])
        self.assertIn("不要写影响", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
