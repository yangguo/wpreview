#!/usr/bin/env python3
"""Excel Structured Reviewer - Read Excel directly and review with LLM."""

import argparse
import json
import os
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import markdown
import openpyxl
from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

STANDARD_SCHEMA_FIELDS = [
    "control_id",
    "control_description",
    "audit_objective",
    "audit_procedure",
    "sample_selection_method",
    "sample_size",
    "test_steps",
    "test_result",
    "exception_flag",
    "conclusion",
]

FIELD_ALIASES = {
    "control_id": ["control id", "控制编号", "控制id", "编号", "id"],
    "control_description": ["control description", "控制描述", "控制要求", "控制活动描述"],
    "audit_objective": ["audit objective", "审计目标", "测试目标"],
    "audit_procedure": ["audit procedure", "审计程序", "测试程序", "审计步骤", "程序要求"],
    "sample_selection_method": ["sample selection method", "抽样方法", "选样方法", "样本抽样依据", "抽样依据"],
    "sample_size": ["sample size", "样本量", "抽样数量", "样本数量"],
    "test_steps": ["test steps", "测试步骤", "执行步骤", "测试过程", "检查步骤"],
    "test_result": ["test result", "测试结果", "执行结果", "检查结果", "结果"],
    "exception_flag": ["exception flag", "是否例外", "例外标记", "异常标记", "缺陷标记"],
    "conclusion": ["conclusion", "结论", "审计结论", "控制结论"],
}


class ExcelImageReviewer:
    """Read Excel structure directly and review workpapers with LLM."""

    def __init__(self, excel_path, output_dir="output", model_name=None, base_url=None):
        self.excel_path = excel_path
        self.output_dir = Path(output_dir)
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sheet_structures = {}
        self.sheet_schema_records = {}
        self.sheet_reviews = {}

        self._workbook = None

        client_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        resolved_url = base_url or os.getenv("OPENAI_BASE_URL")
        if resolved_url:
            client_kwargs["base_url"] = resolved_url
        self.client = OpenAI(**client_kwargs)

    @staticmethod
    def _is_empty(value):
        return value is None or (isinstance(value, str) and value.strip() == "")

    @staticmethod
    def _normalize_text(text):
        if text is None:
            return ""
        normalized = str(text).strip().lower()
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", normalized)

    def _map_schema_field(self, label):
        normalized = self._normalize_text(label)
        if not normalized:
            return None

        for field in STANDARD_SCHEMA_FIELDS:
            if normalized == self._normalize_text(field):
                return field

        for field, aliases in FIELD_ALIASES.items():
            alias_tokens = [self._normalize_text(alias) for alias in aliases]
            if normalized in alias_tokens:
                return field
            if any(alias and (alias in normalized or normalized in alias) for alias in alias_tokens):
                return field
        return None

    def _load_workbook(self):
        if self._workbook is None:
            self._workbook = openpyxl.load_workbook(self.excel_path, data_only=False)
        return self._workbook

    def _detect_table_regions(self, non_empty_positions):
        """Detect contiguous non-empty cell blocks as table regions."""
        unvisited = set(non_empty_positions)
        regions = []

        while unvisited:
            start = min(unvisited)
            queue = deque([start])
            component = set([start])
            unvisited.remove(start)

            while queue:
                row, col = queue.popleft()
                neighbors = [
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ]
                for neighbor in neighbors:
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            rows = [r for r, _ in component]
            cols = [c for _, c in component]
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)
            regions.append(
                {
                    "region_id": f"R{len(regions) + 1}",
                    "min_row": min_row,
                    "max_row": max_row,
                    "min_col": min_col,
                    "max_col": max_col,
                    "cell_count": len(component),
                    "span": f"{openpyxl.utils.get_column_letter(min_col)}{min_row}:{openpyxl.utils.get_column_letter(max_col)}{max_row}",
                }
            )

        regions.sort(key=lambda item: (item["min_row"], item["min_col"]))
        for idx, region in enumerate(regions, start=1):
            region["region_id"] = f"R{idx}"
        return regions

    def extract_sheet_structure(self, sheet_name):
        """Read sheet cells with coordinates, row/col positions, and merged ranges."""
        wb = self._load_workbook()
        ws = wb[sheet_name]

        max_row = ws.max_row or 0
        max_col = ws.max_column or 0
        cells = []
        non_empty_positions = set()

        for row in range(1, max_row + 1):
            for col in range(1, max_col + 1):
                cell = ws.cell(row=row, column=col)
                if self._is_empty(cell.value):
                    continue

                value = str(cell.value)
                cell_record = {
                    "coord": cell.coordinate,
                    "row": row,
                    "column": col,
                    "value": value,
                    "data_type": cell.data_type,
                    "is_formula": value.startswith("="),
                }
                cells.append(cell_record)
                non_empty_positions.add((row, col))

        merged_ranges = []
        for merged in ws.merged_cells.ranges:
            merged_ranges.append(
                {
                    "range": str(merged),
                    "min_row": merged.min_row,
                    "max_row": merged.max_row,
                    "min_col": merged.min_col,
                    "max_col": merged.max_col,
                }
            )

        table_regions = self._detect_table_regions(non_empty_positions)

        structure = {
            "sheet_name": sheet_name,
            "max_row": max_row,
            "max_column": max_col,
            "non_empty_cell_count": len(cells),
            "cells": sorted(cells, key=lambda item: (item["row"], item["column"])),
            "merged_ranges": merged_ranges,
            "table_regions": table_regions,
        }
        self.sheet_structures[sheet_name] = structure
        return structure

    def _extract_from_tabular_region(self, region, cell_map):
        min_row, max_row = region["min_row"], region["max_row"]
        min_col, max_col = region["min_col"], region["max_col"]

        header_row = min_row
        mapped_columns = {}

        for col in range(min_col, max_col + 1):
            label = cell_map.get((header_row, col))
            field = self._map_schema_field(label)
            if field:
                mapped_columns[col] = field

        if len(mapped_columns) < 2:
            return []

        records = []
        for row in range(header_row + 1, max_row + 1):
            record = {field: None for field in STANDARD_SCHEMA_FIELDS}
            source_cells = {}
            has_value = False

            for col, field in mapped_columns.items():
                value = cell_map.get((row, col))
                if not self._is_empty(value):
                    record[field] = value
                    source_cells[field] = f"{openpyxl.utils.get_column_letter(col)}{row}"
                    has_value = True

            if has_value:
                record["sample_id"] = f"{region['region_id']}-{row}"
                record["source_cells"] = source_cells
                records.append(record)

        return records

    def _extract_from_key_value_region(self, region, cell_map):
        min_row, max_row = region["min_row"], region["max_row"]
        min_col, max_col = region["min_col"], region["max_col"]

        record = {field: None for field in STANDARD_SCHEMA_FIELDS}
        source_cells = {}
        mapped_count = 0

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col):
                label = cell_map.get((row, col))
                field = self._map_schema_field(label)
                if not field:
                    continue

                value_col = col + 1
                while value_col <= max_col and self._is_empty(cell_map.get((row, value_col))):
                    value_col += 1
                if value_col <= max_col:
                    value = cell_map.get((row, value_col))
                    if not self._is_empty(value):
                        record[field] = value
                        source_cells[field] = f"{openpyxl.utils.get_column_letter(value_col)}{row}"
                        mapped_count += 1

        if mapped_count < 2:
            return []

        record["sample_id"] = f"{region['region_id']}-KV"
        record["source_cells"] = source_cells
        return [record]

    def map_sheet_to_schema(self, sheet_name):
        """Map extracted table content to the standard audit workpaper schema."""
        if sheet_name not in self.sheet_structures:
            self.extract_sheet_structure(sheet_name)

        structure = self.sheet_structures[sheet_name]
        cell_map = {
            (cell["row"], cell["column"]): cell["value"]
            for cell in structure["cells"]
        }

        schema_records = []
        for region in structure["table_regions"]:
            tabular_records = self._extract_from_tabular_region(region, cell_map)
            if tabular_records:
                schema_records.extend(tabular_records)
                continue

            kv_records = self._extract_from_key_value_region(region, cell_map)
            if kv_records:
                schema_records.extend(kv_records)

        self.sheet_schema_records[sheet_name] = schema_records
        return schema_records

    def _build_sheet_review_payload(self, sheet_name):
        structure = self.sheet_structures.get(sheet_name, {})
        schema_records = self.sheet_schema_records.get(sheet_name, [])

        return {
            "sheet_name": sheet_name,
            "standard_schema": STANDARD_SCHEMA_FIELDS,
            "structure_summary": {
                "max_row": structure.get("max_row", 0),
                "max_column": structure.get("max_column", 0),
                "non_empty_cell_count": structure.get("non_empty_cell_count", 0),
                "merged_ranges": structure.get("merged_ranges", []),
                "table_regions": structure.get("table_regions", []),
            },
            "schema_records": schema_records,
            "cell_excerpt": structure.get("cells", [])[:200],
        }

    def review_structured_sheet(self, sheet_name):
        """Review a sheet using structured Excel content (no image recognition)."""
        payload = self._build_sheet_review_payload(sheet_name)

        prompt = (
            "你是IT审计底稿审阅专家。输入数据来自 openpyxl 对 Excel 的结构化读取，不是截图。\n"
            "请按以下四个维度审阅：\n"
            "A. 覆盖性：测试过程是否覆盖审计程序要求，输出 coverage_status（完整/部分覆盖/未覆盖）、missing_points、risk_impact。\n"
            "B. 方法性问题：识别如仅询问无证据、只测设计不测执行、抽样依据不清、样本量不足、无例外闭环、证据与结论不匹配。\n"
            "C. 逻辑问题（内部自洽）：测试结果、结论、步骤之间矛盾。\n"
            "D. 跨字段一致性：控制描述/审计程序/测试过程/测试结果/结论/例外标记/期间与样本日期。\n\n"
            "输出必须为中文 Markdown，严格使用以下结构：\n"
            "## 审阅总览\n"
            "- coverage_status: \n"
            "- missing_points: \n"
            "- risk_impact: \n\n"
            "## 问题清单\n"
            "|问题ID|问题类型|严重级别|定位（底稿字段/样本编号）|原文摘录|判定依据|整改建议|\n"
            "|---|---|---|---|---|---|---|\n"
            "至少输出1行；如果无问题，输出“未发现重大问题”，并给出“建议补充检查项”。\n\n"
            "## 需补充证据\n"
            "- 列出无法直接从当前底稿判断但影响结论可靠性的证据缺口。\n\n"
            "定位必须引用字段名或 sample_id；原文摘录必须来自输入内容，不可编造。"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是严谨的IT审计底稿审阅专家，请始终使用中文 Markdown。",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"审阅对象: {sheet_name}\n\n"
                            f"审阅要求:\n{prompt}\n\n"
                            f"结构化输入(JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                        ),
                    },
                ],
                max_tokens=2200,
                temperature=0.2,
                timeout=180,
            )
            return response.choices[0].message.content
        except Exception as exc:
            return f"Error reviewing structured sheet: {exc}"

    # Backward-compatible alias for callers still using the old method name.
    def review_image(self, image, sheet_name):
        del image
        return self.review_structured_sheet(sheet_name)

    def _add_inline_runs(self, paragraph, node):
        """Append HTML inline nodes into a docx paragraph while preserving simple styles."""
        if isinstance(node, NavigableString):
            text = str(node)
            if text:
                paragraph.add_run(text)
            return
        if not isinstance(node, Tag):
            return

        if node.name == "br":
            paragraph.add_run("\n")
            return

        if node.name in ("strong", "b"):
            run = paragraph.add_run(node.get_text())
            run.bold = True
            return
        if node.name in ("em", "i"):
            run = paragraph.add_run(node.get_text())
            run.italic = True
            return
        if node.name == "code":
            run = paragraph.add_run(node.get_text())
            run.font.name = "Courier New"
            return
        if node.name == "a":
            text = node.get_text()
            href = node.get("href", "")
            paragraph.add_run(f"{text} ({href})" if href else text)
            return

        for child in node.children:
            self._add_inline_runs(paragraph, child)

    def _append_markdown_to_doc(self, doc, markdown_text):
        """Render markdown text into a Word document."""
        html = markdown.markdown(markdown_text or "", extensions=["fenced_code", "tables", "sane_lists"])
        soup = BeautifulSoup(html, "html.parser")

        def render_block(tag):
            if tag.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = min(6, int(tag.name[1]))
                p = doc.add_heading(level=level)
                for child in tag.children:
                    self._add_inline_runs(p, child)
                return

            if tag.name == "p":
                p = doc.add_paragraph()
                for child in tag.children:
                    self._add_inline_runs(p, child)
                return

            if tag.name in ("ul", "ol"):
                style = "List Bullet" if tag.name == "ul" else "List Number"
                for li in tag.find_all("li", recursive=False):
                    p = doc.add_paragraph(style=style)
                    for child in li.children:
                        self._add_inline_runs(p, child)
                return

            if tag.name == "pre":
                p = doc.add_paragraph()
                run = p.add_run(tag.get_text())
                run.font.name = "Courier New"
                return

            if tag.name == "blockquote":
                p = doc.add_paragraph()
                p.style = "Intense Quote"
                p.add_run(tag.get_text())
                return

            if tag.name == "table":
                rows = tag.find_all("tr")
                if not rows:
                    return
                col_count = max(len(r.find_all(["th", "td"])) for r in rows)
                table = doc.add_table(rows=0, cols=col_count)
                table.style = "Table Grid"
                for row_tag in rows:
                    cells = row_tag.find_all(["th", "td"])
                    row_cells = table.add_row().cells
                    for idx, cell in enumerate(cells):
                        row_cells[idx].text = re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
                return

            text = tag.get_text(" ", strip=True)
            if text:
                doc.add_paragraph(text)

        for elem in soup.contents:
            if isinstance(elem, NavigableString):
                raw = str(elem).strip()
                if raw:
                    doc.add_paragraph(raw)
                continue
            if isinstance(elem, Tag):
                render_block(elem)

    def _save_sheet_payload_json(self, sheet_name):
        payload = self._build_sheet_review_payload(sheet_name)
        payload_path = self.output_dir / f"{sheet_name}_structured.json"
        with open(payload_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return payload_path

    def process_excel(self, sheets=None, limit=None):
        """Read sheet structures, map schema fields, and review each sheet with LLM."""
        print(f"Processing: {self.excel_path}")

        wb = self._load_workbook()
        all_sheet_names = wb.sheetnames
        print(f"Found {len(all_sheet_names)} sheet(s): {', '.join(all_sheet_names)}")

        if sheets:
            sheet_names = [name for name in all_sheet_names if name in sheets]
        else:
            sheet_names = all_sheet_names

        if limit:
            sheet_names = sheet_names[:limit]

        if len(sheet_names) < len(all_sheet_names):
            print(f"Processing {len(sheet_names)} sheet(s): {', '.join(sheet_names)}")

        for sheet_name in sheet_names:
            print(f"\n[{sheet_name}] Extracting structure...")
            try:
                self.extract_sheet_structure(sheet_name)
                self.map_sheet_to_schema(sheet_name)
                payload_path = self._save_sheet_payload_json(sheet_name)
                print(f"[{sheet_name}] Structured payload saved: {payload_path}")
                print(f"[{sheet_name}] Reviewing with LLM...")
                self.sheet_reviews[sheet_name] = self.review_structured_sheet(sheet_name)
                print(f"[{sheet_name}] Done.")
            except Exception as exc:
                print(f"[{sheet_name}] Error: {exc}")
                self.sheet_reviews[sheet_name] = f"Error: {exc}"

    def _append_schema_preview_table(self, doc, schema_records):
        if not schema_records:
            doc.add_paragraph("未抽取到可映射标准 Schema 的记录。")
            return

        preview = schema_records[:8]
        columns = ["sample_id"] + STANDARD_SCHEMA_FIELDS
        table = doc.add_table(rows=1, cols=len(columns))
        table.style = "Table Grid"
        for idx, col in enumerate(columns):
            table.rows[0].cells[idx].text = col

        for record in preview:
            row_cells = table.add_row().cells
            for idx, col in enumerate(columns):
                row_cells[idx].text = str(record.get(col, "") or "")

        if len(schema_records) > len(preview):
            doc.add_paragraph(f"仅展示前 {len(preview)} 条，共 {len(schema_records)} 条。")

    def generate_report(self):
        """Generate a DOCX report based on structured extraction and LLM results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc = Document()
        doc.add_heading("Excel 底稿结构化审阅报告", level=1)
        doc.add_paragraph(f"文件：{os.path.basename(self.excel_path)}")
        doc.add_paragraph(f"生成时间：{timestamp}")
        doc.add_paragraph(f"审阅工作表数量：{len(self.sheet_reviews)}")
        doc.add_paragraph(
            "说明：本报告基于 openpyxl 直接读取单元格坐标、行列位置、合并信息与 Schema 映射结果，再由 LLM 完成覆盖性/方法性/逻辑/一致性审阅。"
        )

        for sheet_name in self.sheet_reviews.keys():
            structure = self.sheet_structures.get(sheet_name, {})
            schema_records = self.sheet_schema_records.get(sheet_name, [])
            review = self.sheet_reviews.get(sheet_name, "未生成审阅意见。")

            doc.add_heading(f"工作表：{sheet_name}", level=2)
            doc.add_paragraph(
                f"非空单元格：{structure.get('non_empty_cell_count', 0)}；"
                f"表格区域：{len(structure.get('table_regions', []))}；"
                f"合并单元格区域：{len(structure.get('merged_ranges', []))}；"
                f"Schema记录：{len(schema_records)}"
            )

            merged_ranges = structure.get("merged_ranges", [])
            if merged_ranges:
                merged_preview = ", ".join(item["range"] for item in merged_ranges[:10])
                doc.add_paragraph(f"合并区域（预览）：{merged_preview}")

            doc.add_heading("标准 Schema 抽取结果（预览）", level=3)
            self._append_schema_preview_table(doc, schema_records)

            doc.add_heading("审阅结果", level=3)
            self._append_markdown_to_doc(doc, review)

        report_path = self.output_dir / "review_report.docx"
        doc.save(str(report_path))
        print(f"Report saved: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Read Excel structure directly and review with LLM"
    )
    parser.add_argument("excel_file", help="Path to the Excel file")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("-m", "--model", default=None, help="Model to use (default: OPENAI_MODEL env or gpt-4o)")
    parser.add_argument("-u", "--url", default=None, help="Base URL for OpenAI-compatible API endpoint (default: OPENAI_BASE_URL env)")
    parser.add_argument("-s", "--sheets", nargs="+", default=None, help="Specific sheet names to process")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Limit to first N sheets")
    args = parser.parse_args()

    if not os.path.exists(args.excel_file):
        print(f"Error: File not found: {args.excel_file}")
        sys.exit(1)
    if not args.excel_file.endswith((".xlsx", ".xls")):
        print("Error: File must be an Excel file (.xlsx or .xls)")
        sys.exit(1)

    reviewer = ExcelImageReviewer(args.excel_file, args.output, args.model, args.url)
    reviewer.process_excel(sheets=args.sheets, limit=args.limit)
    report_path = reviewer.generate_report()
    print(f"\nDone! Report: {report_path}")


if __name__ == "__main__":
    main()
