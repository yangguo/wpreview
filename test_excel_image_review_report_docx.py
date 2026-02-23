#!/usr/bin/env python3
"""Tests for structured review prompt and DOCX report generation."""

import tempfile
import unittest
from pathlib import Path

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

    def test_generate_report_outputs_docx_and_includes_schema_preview(self):
        reviewer = self._reviewer()

        reviewer.sheet_structures = {
            "Sheet1": {
                "non_empty_cell_count": 10,
                "table_regions": [{"region_id": "R1"}],
                "merged_ranges": [{"range": "A1:B1"}],
            }
        }
        reviewer.sheet_schema_records = {
            "Sheet1": [
                {
                    "sample_id": "R1-2",
                    "control_id": "CTRL-1",
                    "control_description": "权限审批控制",
                    "audit_objective": "验证审批有效性",
                    "audit_procedure": "抽样检查审批与日志",
                    "sample_selection_method": "判断抽样",
                    "sample_size": "5",
                    "test_steps": "核对审批流",
                    "test_result": "2笔缺审批",
                    "exception_flag": "Y",
                    "conclusion": "控制需改进",
                }
            ]
        }
        reviewer.sheet_reviews = {
            "Sheet1": (
                "## 审阅总览\n"
                "- coverage_status: 部分覆盖\n"
                "- missing_points: 未追踪例外闭环\n"
                "- risk_impact: 可能影响结论可靠性\n\n"
                "## 问题清单\n"
                "|问题ID|问题类型|严重级别|定位（底稿字段/样本编号）|原文摘录|判定依据|整改建议|\n"
                "|---|---|---|---|---|---|---|\n"
                "|Q1|方法性问题|高|test_steps / R1-2|仅核对审批流|缺少例外追踪|补充闭环检查|"
            )
        }

        report_path = reviewer.generate_report()

        self.assertTrue(Path(report_path).exists())
        self.assertEqual(Path(report_path).suffix, ".docx")

        doc = Document(str(report_path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        self.assertIn("Excel 底稿结构化审阅报告", text)
        self.assertIn("标准 Schema 抽取结果", text)
        self.assertIn("审阅总览", text)
        self.assertIn("coverage_status", text)

    def test_review_prompt_requires_coverage_methodology_logic_consistency(self):
        reviewer = self._reviewer()
        reviewer.sheet_structures = {
            "Sheet1": {
                "max_row": 3,
                "max_column": 3,
                "non_empty_cell_count": 3,
                "cells": [{"coord": "A1", "row": 1, "column": 1, "value": "control_id"}],
                "merged_ranges": [],
                "table_regions": [{"region_id": "R1", "min_row": 1, "max_row": 3, "min_col": 1, "max_col": 3}],
            }
        }
        reviewer.sheet_schema_records = {
            "Sheet1": [
                {
                    "sample_id": "R1-2",
                    "control_id": "CTRL-1",
                    "audit_procedure": "抽样检查审批、复核日志、追踪异常处理",
                    "test_steps": "抽样检查审批",
                    "test_result": "未见异常",
                    "conclusion": "控制有效",
                    "source_cells": {"control_id": "A2"},
                }
            ]
        }

        captured = {"prompt": ""}

        class FakeResponse:
            class Choice:
                class Message:
                    content = "ok"

                message = Message()

            choices = [Choice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured["prompt"] = kwargs["messages"][1]["content"]
                return FakeResponse()

        reviewer.client = type(
            "FakeClient",
            (),
            {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
        )()

        reviewer.review_structured_sheet("Sheet1")
        self.assertIn("覆盖性", captured["prompt"])
        self.assertIn("方法性问题", captured["prompt"])
        self.assertIn("逻辑问题", captured["prompt"])
        self.assertIn("跨字段一致性", captured["prompt"])
        self.assertIn("coverage_status", captured["prompt"])
        self.assertIn("missing_points", captured["prompt"])
        self.assertIn("risk_impact", captured["prompt"])
        self.assertIn("问题清单", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
