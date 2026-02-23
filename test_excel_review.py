#!/usr/bin/env python3
"""End-to-end test for structured Excel workpaper review (mock LLM)."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from docx import Document


class MockCompletion:
    class Choice:
        class Message:
            def __init__(self):
                self.content = (
                    "## 审阅总览\n"
                    "- coverage_status: 部分覆盖\n"
                    "- missing_points: 未覆盖异常闭环\n"
                    "- risk_impact: 中\n\n"
                    "## 问题清单\n"
                    "|问题ID|问题类型|严重级别|定位（底稿字段/样本编号）|原文摘录|判定依据|整改建议|\n"
                    "|---|---|---|---|---|---|---|\n"
                    "|Q1|方法性问题|中|test_steps / R1-2|仅完成询问|缺少证据检查|补充日志与凭证检查|\n\n"
                    "## 需补充证据\n"
                    "- 缺少样本抽取依据文档。"
                )

        def __init__(self):
            self.message = self.Message()

    def __init__(self):
        self.choices = [self.Choice()]


class MockChatCompletions:
    def create(self, **kwargs):
        return MockCompletion()


class MockClient:
    def __init__(self, **kwargs):
        self.chat = type("obj", (object,), {"completions": MockChatCompletions()})()


sys.modules["openai"] = MagicMock()
sys.modules["openai"].OpenAI = MockClient
sys.modules["openai"].AzureOpenAI = MockClient

from excel_image_review import ExcelImageReviewer


class EndToEndStructuredReviewTests(unittest.TestCase):
    def test_full_workflow_generates_structured_payload_and_docx_report(self):
        if not os.path.exists("sample_data.xlsx"):
            subprocess.run([sys.executable, "create_sample_excel.py"], check=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            reviewer = ExcelImageReviewer("sample_data.xlsx", tmp_dir, model_name="gpt-4o")
            reviewer.process_excel(limit=1)
            report_path = reviewer.generate_report()

            self.assertTrue(reviewer.sheet_structures)
            self.assertTrue(reviewer.sheet_schema_records is not None)
            self.assertTrue(reviewer.sheet_reviews)

            payload_files = list(Path(tmp_dir).glob("*_structured.json"))
            self.assertTrue(payload_files, "Expected structured JSON payload output")

            self.assertTrue(Path(report_path).exists())
            self.assertEqual(Path(report_path).suffix, ".docx")

            doc = Document(str(report_path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            self.assertIn("Excel 底稿结构化审阅报告", text)
            self.assertIn("审阅结果", text)
            self.assertIn("审阅总览", text)


if __name__ == "__main__":
    unittest.main()
