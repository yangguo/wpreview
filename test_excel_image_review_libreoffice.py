#!/usr/bin/env python3
"""Unit tests for structured Excel extraction and schema mapping."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import Workbook


class DummyOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StructuredExtractionTests(unittest.TestCase):
    def setUp(self):
        self.work_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.work_dir.cleanup)
        self.excel_path = Path(self.work_dir.name) / "book.xlsx"

        wb = Workbook()

        ws1 = wb.active
        ws1.title = "Tabular"
        ws1.append(["control_id", "audit_procedure", "test_steps", "test_result", "conclusion"])
        ws1.append(["CTRL-001", "抽样检查审批", "核对审批流并复核日志", "发现1个例外", "控制基本有效"])
        ws1.merge_cells("A4:B4")
        ws1["A4"] = "Merged Note"

        ws2 = wb.create_sheet("KeyValue")
        ws2.append(["控制编号", "CTRL-100"])
        ws2.append(["审计程序", "检查审批、复核日志、追踪异常处理"])
        ws2.append(["测试步骤", "抽样2个月交易并追踪例外整改"])
        ws2.append(["测试结果", "发现审批缺失2笔"])
        ws2.append(["结论", "控制无效"])

        wb.save(self.excel_path)

    def _build_reviewer(self):
        from excel_image_review import ExcelImageReviewer

        return ExcelImageReviewer(str(self.excel_path), output_dir=self.work_dir.name)

    @patch("excel_image_review.OpenAI", new=DummyOpenAIClient)
    def test_extract_sheet_structure_preserves_coordinates_and_merged_ranges(self):
        reviewer = self._build_reviewer()
        structure = reviewer.extract_sheet_structure("Tabular")

        self.assertEqual(structure["sheet_name"], "Tabular")
        self.assertGreater(structure["non_empty_cell_count"], 0)
        self.assertTrue(any(cell["coord"] == "A2" for cell in structure["cells"]))
        self.assertTrue(any(item["range"] == "A4:B4" for item in structure["merged_ranges"]))
        self.assertTrue(structure["table_regions"])

    @patch("excel_image_review.OpenAI", new=DummyOpenAIClient)
    def test_map_sheet_to_schema_from_tabular_headers(self):
        reviewer = self._build_reviewer()
        reviewer.extract_sheet_structure("Tabular")
        records = reviewer.map_sheet_to_schema("Tabular")

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["control_id"], "CTRL-001")
        self.assertEqual(record["audit_procedure"], "抽样检查审批")
        self.assertEqual(record["test_steps"], "核对审批流并复核日志")
        self.assertEqual(record["test_result"], "发现1个例外")
        self.assertEqual(record["conclusion"], "控制基本有效")
        self.assertEqual(record["source_cells"]["control_id"], "A2")

    @patch("excel_image_review.OpenAI", new=DummyOpenAIClient)
    def test_map_sheet_to_schema_from_key_value_layout(self):
        reviewer = self._build_reviewer()
        reviewer.extract_sheet_structure("KeyValue")
        records = reviewer.map_sheet_to_schema("KeyValue")

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["control_id"], "CTRL-100")
        self.assertIn("复核日志", record["audit_procedure"])
        self.assertIn("审批缺失", record["test_result"])
        self.assertEqual(record["conclusion"], "控制无效")


if __name__ == "__main__":
    unittest.main()
