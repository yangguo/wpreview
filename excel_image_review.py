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
    "control_description": ["control description", "控制描述", "控制要求", "控制活动描述", "控制活动"],
    "audit_objective": ["audit objective", "审计目标", "测试目标"],
    "audit_procedure": ["audit procedure", "审计程序", "测试程序", "审计步骤", "程序要求", "标准审计程序", "执行的审计程序"],
    "sample_selection_method": ["sample selection method", "抽样方法", "选样方法", "样本抽样依据", "抽样依据", "控制类型", "发生频率"],
    "sample_size": ["sample size", "样本量", "抽样数量", "样本数量", "样本总量", "测试期间样本总量", "测试期间样本量"],
    "test_steps": ["test steps", "测试步骤", "执行步骤", "测试过程", "检查步骤"],
    "test_result": ["test result", "测试结果", "执行结果", "检查结果", "结果", "设计有效性测试结论", "设计有效性结论"],
    "exception_flag": ["exception flag", "是否例外", "例外标记", "异常标记", "缺陷标记", "是否发现异常", "是否异常"],
    "conclusion": ["conclusion", "结论", "审计结论", "控制结论", "执行有效性测试结论", "执行有效性结论"],
}

CONTEXT_CARRY_FIELDS = [
    "control_id",
    "control_description",
    "audit_objective",
    "audit_procedure",
    "sample_selection_method",
]

# Only longer aliases use fuzzy "contains" matching to avoid false positives
# from short tokens like "id" or "结果".
MIN_FUZZY_ALIAS_LEN = 3

# Domain-specific phrases used to infer a positive/clean result in Q&A layouts.
QA_POSITIVE_RESULT_HINTS = ("未见异常", "无异常")


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
            if any(alias and len(alias) >= MIN_FUZZY_ALIAS_LEN and (alias in normalized) for alias in alias_tokens):
                return field
        return None

    def _resolve_formula_reference(self, value, depth=0):
        if value is None:
            return None
        if not isinstance(value, str):
            return str(value)

        text = value.strip()
        if not text.startswith("="):
            return value
        if depth >= 3:
            return value

        match = re.match(r"^=\s*(?:'((?:[^']|'')+)'|([^'!]+))!([$]?[A-Za-z]+[$]?\d+)$", text)
        if not match:
            return value

        sheet_name = (match.group(1) or match.group(2) or "").replace("''", "'")
        coord = match.group(3).replace("$", "")
        try:
            wb = self._load_workbook()
            target_ws = wb[sheet_name]
            target_value = target_ws[coord].value
        except (KeyError, ValueError, TypeError):
            return value

        if self._is_empty(target_value):
            return None
        if isinstance(target_value, str) and target_value.strip().startswith("="):
            return self._resolve_formula_reference(target_value, depth + 1)
        return str(target_value)

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

                value = self._resolve_formula_reference(cell.value)
                if self._is_empty(value):
                    continue
                value = str(value)
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

        header_row = None
        mapped_columns = {}
        anchor_fields = {"control_id", "control_description", "test_result", "conclusion", "exception_flag"}
        best_score = 0

        for row in range(min_row, max_row + 1):
            row_mapping = {}
            used_fields = set()
            for col in range(min_col, max_col + 1):
                label = cell_map.get((row, col))
                field = self._map_schema_field(label)
                if field and field not in used_fields:
                    row_mapping[col] = field
                    used_fields.add(field)
            score = len(row_mapping)
            if score < 2:
                continue
            if not (used_fields & anchor_fields):
                continue
            if score > best_score:
                header_row = row
                mapped_columns = row_mapping
                best_score = score

        if not header_row or len(mapped_columns) < 2:
            return []

        records = []
        row_context = {}
        row_context_cells = {}
        for row in range(header_row + 1, max_row + 1):
            record = {field: None for field in STANDARD_SCHEMA_FIELDS}
            source_cells = {}
            has_direct_value = False

            for col, field in mapped_columns.items():
                value = cell_map.get((row, col))
                if not self._is_empty(value):
                    record[field] = value
                    source_cells[field] = f"{openpyxl.utils.get_column_letter(col)}{row}"
                    has_direct_value = True

            if not has_direct_value:
                continue

            for field in CONTEXT_CARRY_FIELDS:
                if self._is_empty(record[field]) and field in row_context:
                    record[field] = row_context[field]
                    if field in row_context_cells and field not in source_cells:
                        source_cells[field] = row_context_cells[field]

            for field in CONTEXT_CARRY_FIELDS:
                if not self._is_empty(record[field]):
                    row_context[field] = record[field]
                    if field in source_cells:
                        row_context_cells[field] = source_cells[field]

            record["sample_id"] = f"{region['region_id']}-{row}"
            record["source_cells"] = source_cells
            records.append(record)

        return records

    def _extract_from_key_value_region(self, region, cell_map, sheet_max_col=None):
        min_row, max_row = region["min_row"], region["max_row"]
        min_col, max_col = region["min_col"], region["max_col"]
        max_search_col = max_col if sheet_max_col is None else max(max_col, sheet_max_col)

        record = {field: None for field in STANDARD_SCHEMA_FIELDS}
        source_cells = {}
        mapped_fields = set()

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                label = cell_map.get((row, col))
                field = self._map_schema_field(label)
                if not field:
                    continue

                value_col = col + 1
                while value_col <= max_search_col and self._is_empty(cell_map.get((row, value_col))):
                    value_col += 1
                if value_col <= max_search_col:
                    value = cell_map.get((row, value_col))
                    if not self._is_empty(value):
                        record[field] = value
                        source_cells[field] = f"{openpyxl.utils.get_column_letter(value_col)}{row}"
                        mapped_fields.add(field)

        mapped_count = len(mapped_fields)
        if mapped_count < 1:
            return []
        # Some workpapers split labels into tiny one-line KV regions; allow
        # single-field anchor records so downstream context carry can stitch
        # surrounding metadata together.
        if mapped_count == 1 and not (
            mapped_fields
            & {"control_id", "control_description", "test_result", "exception_flag", "conclusion"}
        ):
            return []

        record["sample_id"] = f"{region['region_id']}-KV"
        record["source_cells"] = source_cells
        return [record]

    def _extract_from_step_block_region(self, region, cell_map):
        min_row, max_row = region["min_row"], region["max_row"]
        min_col, max_col = region["min_col"], region["max_col"]

        header_row = None
        step_col = None
        standard_col = None
        execution_cols = []

        for row in range(min_row, max_row + 1):
            row_step_col = None
            row_standard_col = None
            row_execution_cols = []

            for col in range(min_col, max_col + 1):
                label = str(cell_map.get((row, col)) or "").strip()
                normalized = self._normalize_text(label)
                if not normalized:
                    continue

                if "测试步骤" in normalized:
                    row_step_col = col
                if "标准审计程序" in normalized:
                    row_standard_col = col
                if "执行" in normalized and "审计程序" in normalized and "标准" not in normalized:
                    row_execution_cols.append(col)

            if row_step_col and row_standard_col and row_execution_cols:
                header_row = row
                step_col = row_step_col
                standard_col = row_standard_col
                execution_cols = sorted(set(row_execution_cols))
                break

        if not header_row or not standard_col or not execution_cols:
            return []

        conclusion_row = None
        detail_rows = []
        for row in range(header_row + 1, max_row + 1):
            marker = str(cell_map.get((row, step_col)) or "").strip()
            marker_normalized = self._normalize_text(marker)
            if "测试结论" in marker_normalized:
                conclusion_row = row
                break

            standard_text = str(cell_map.get((row, standard_col)) or "").strip()
            execution_has_value = any(str(cell_map.get((row, col)) or "").strip() for col in execution_cols)
            if standard_text or execution_has_value:
                detail_rows.append(row)

        if not detail_rows:
            return []

        standard_text_lines = []
        standard_source_cells = []
        for row in detail_rows:
            standard_text = str(cell_map.get((row, standard_col)) or "").strip()
            if not standard_text:
                continue
            marker = str(cell_map.get((row, step_col)) or "").strip()
            if marker.isdigit():
                standard_text_lines.append(f"{marker}. {standard_text}")
            else:
                standard_text_lines.append(standard_text)
            standard_source_cells.append(f"{openpyxl.utils.get_column_letter(standard_col)}{row}")

        records = []
        for col in execution_cols:
            exec_text_lines = []
            exec_source_cells = []

            for row in detail_rows:
                exec_text = str(cell_map.get((row, col)) or "").strip()
                if not exec_text:
                    continue
                marker = str(cell_map.get((row, step_col)) or "").strip()
                if marker.isdigit():
                    exec_text_lines.append(f"{marker}. {exec_text}")
                else:
                    exec_text_lines.append(exec_text)
                exec_source_cells.append(f"{openpyxl.utils.get_column_letter(col)}{row}")

            conclusion_value = None
            conclusion_cell = None
            if conclusion_row:
                maybe_conclusion = str(cell_map.get((conclusion_row, col)) or "").strip()
                if maybe_conclusion:
                    conclusion_value = maybe_conclusion
                    conclusion_cell = f"{openpyxl.utils.get_column_letter(col)}{conclusion_row}"

            if not exec_text_lines and not conclusion_value:
                continue

            record = {field: None for field in STANDARD_SCHEMA_FIELDS}
            if standard_text_lines:
                record["audit_procedure"] = "\n".join(standard_text_lines)
            if exec_text_lines:
                record["test_steps"] = "\n".join(exec_text_lines)
            if conclusion_value:
                record["conclusion"] = conclusion_value

            source_cells = {}
            if standard_source_cells:
                source_cells["audit_procedure"] = ",".join(standard_source_cells)
            if exec_source_cells:
                source_cells["test_steps"] = ",".join(exec_source_cells)
            if conclusion_cell:
                source_cells["conclusion"] = conclusion_cell

            record["source_cells"] = source_cells
            record["sample_id"] = f"{region['region_id']}-STEP-{openpyxl.utils.get_column_letter(col)}"
            records.append(record)

        return records

    def _extract_from_procedure_pair_layout(self, structure, cell_map):
        """Extract row-level standard-vs-execution procedure pairs across the full sheet."""
        max_row = structure.get("max_row", 0)
        max_col = structure.get("max_column", 0)
        records = []
        seen_pairs = set()

        for header_row in range(1, max_row + 1):
            standard_cols = []
            execution_cols = []

            for col in range(1, max_col + 1):
                normalized = self._normalize_text(cell_map.get((header_row, col)))
                if not normalized:
                    continue
                if "标准审计程序" in normalized:
                    standard_cols.append(col)
                if "执行" in normalized and "审计程序" in normalized and "标准" not in normalized:
                    execution_cols.append(col)

            if not standard_cols or not execution_cols:
                continue

            for standard_col in standard_cols:
                right_execution_cols = [col for col in execution_cols if col > standard_col]
                if right_execution_cols:
                    execution_col = min(right_execution_cols)
                else:
                    execution_col = min(execution_cols, key=lambda col: abs(col - standard_col))

                for row in range(header_row + 1, max_row + 1):
                    standard_text = str(cell_map.get((row, standard_col)) or "").strip()
                    execution_text = str(cell_map.get((row, execution_col)) or "").strip()

                    marker_text = ""
                    if standard_col > 1:
                        marker_text = str(cell_map.get((row, standard_col - 1)) or "").strip()

                    marker_normalized = self._normalize_text(marker_text)
                    standard_normalized = self._normalize_text(standard_text)
                    if any(token in marker_normalized for token in ("测试结论", "样本记录", "缺陷评估")):
                        break
                    if any(token in standard_normalized for token in ("测试结论", "样本记录", "缺陷评估")):
                        break

                    if not standard_text and not execution_text:
                        continue

                    if any(token in standard_normalized for token in ("审计证据", "样本总量", "抽样数量", "审计期间")):
                        continue
                    if any(token in standard_normalized for token in ("测试步骤", "标准审计程序")):
                        continue
                    if "执行的审计程序" in self._normalize_text(execution_text):
                        continue

                    if not standard_text or not execution_text:
                        continue

                    standard_coord = f"{openpyxl.utils.get_column_letter(standard_col)}{row}"
                    execution_coord = f"{openpyxl.utils.get_column_letter(execution_col)}{row}"
                    pair_key = (standard_coord, execution_coord)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    if marker_text.isdigit():
                        standard_value = f"{marker_text}. {standard_text}"
                        execution_value = f"{marker_text}. {execution_text}"
                    else:
                        standard_value = standard_text
                        execution_value = execution_text

                    record = {field: None for field in STANDARD_SCHEMA_FIELDS}
                    record["sample_id"] = f"R0-PAIR-{openpyxl.utils.get_column_letter(execution_col)}{row}"
                    record["audit_procedure"] = standard_value
                    record["test_steps"] = execution_value
                    record["source_cells"] = {
                        "audit_procedure": standard_coord,
                        "test_steps": execution_coord,
                    }
                    records.append(record)

        return records

    def _extract_from_question_answer_layout(self, structure, cell_map):
        max_row = structure.get("max_row", 0)
        max_col = structure.get("max_column", 0)
        qa_pairs = []

        for row in range(1, max_row + 1):
            row_values = []
            for col in range(1, max_col + 1):
                value = cell_map.get((row, col))
                if self._is_empty(value):
                    continue
                row_values.append((col, str(value).strip()))

            if len(row_values) < 2:
                continue

            question_col, question = row_values[0]
            answer_col, answer = row_values[1]
            if self._is_empty(question) or self._is_empty(answer):
                continue
            if question in {"问题", "question"} and answer in {"回答", "answer"}:
                continue

            qa_pairs.append((row, question_col, question, answer_col, answer))

        if not qa_pairs:
            return []

        record = {field: None for field in STANDARD_SCHEMA_FIELDS}
        record["sample_id"] = "R0-QA"
        record["audit_procedure"] = "\n".join(
            f"{idx}. {question}" for idx, (_, _, question, _, _) in enumerate(qa_pairs, start=1)
        )
        record["test_steps"] = "\n".join(
            f"{idx}. {answer}" for idx, (_, _, _, _, answer) in enumerate(qa_pairs, start=1)
        )

        if any(any(hint in answer for hint in QA_POSITIVE_RESULT_HINTS) for _, _, _, _, answer in qa_pairs):
            record["test_result"] = "未见异常"

        question_cells = []
        answer_cells = []
        for row, question_col, _, answer_col, _ in qa_pairs:
            question_cells.append(f"{openpyxl.utils.get_column_letter(question_col)}{row}")
            answer_cells.append(f"{openpyxl.utils.get_column_letter(answer_col)}{row}")

        record["source_cells"] = {
            "audit_procedure": ",".join(question_cells),
            "test_steps": ",".join(answer_cells),
        }
        if record["test_result"]:
            record["source_cells"]["test_result"] = ",".join(answer_cells)

        return [record]

    @staticmethod
    def _extract_coords(coord_value):
        if coord_value is None:
            return []
        raw = str(coord_value)
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
        coords = []
        for token in tokens:
            match = re.match(r"^([A-Za-z]+)(\d+)$", token)
            if not match:
                continue
            col = openpyxl.utils.column_index_from_string(match.group(1).upper())
            row = int(match.group(2))
            coords.append((row, col))
        return coords

    def _record_sort_key(self, record):
        source_cells = record.get("source_cells", {}) or {}
        coord_candidates = []

        for field in ["audit_procedure", "test_steps", "sample_size", "conclusion", "test_result", "exception_flag"]:
            coord_candidates.extend(self._extract_coords(source_cells.get(field)))

        if not coord_candidates:
            for value in source_cells.values():
                coord_candidates.extend(self._extract_coords(value))

        if coord_candidates:
            row, col = min(coord_candidates)
            return (row, col, str(record.get("sample_id") or ""))

        sample_id = str(record.get("sample_id") or "")
        sample_row_match = re.search(r"-(\d+)$", sample_id)
        fallback_row = int(sample_row_match.group(1)) if sample_row_match else 10**9
        return (fallback_row, 10**9, sample_id)

    def _prune_incomplete_procedure_rows(self, schema_records):
        if not schema_records:
            return schema_records

        has_complete_pair = any(
            not self._is_empty(record.get("audit_procedure")) and not self._is_empty(record.get("test_steps"))
            for record in schema_records
        )
        if not has_complete_pair:
            return schema_records

        return [
            record
            for record in schema_records
            if not self._is_empty(record.get("audit_procedure")) and not self._is_empty(record.get("test_steps"))
        ]

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

            step_records = self._extract_from_step_block_region(region, cell_map)
            if step_records:
                schema_records.extend(step_records)

            kv_records = self._extract_from_key_value_region(
                region,
                cell_map,
                sheet_max_col=structure.get("max_column", region["max_col"]),
            )
            if kv_records:
                schema_records.extend(kv_records)

        schema_records.extend(self._extract_from_procedure_pair_layout(structure, cell_map))

        context_values = {}
        context_cells = {}
        for record in schema_records:
            source_cells = record.setdefault("source_cells", {})
            for field in CONTEXT_CARRY_FIELDS:
                value = record.get(field)
                if self._is_empty(value):
                    if field in context_values:
                        record[field] = context_values[field]
                        if field in context_cells and field not in source_cells:
                            source_cells[field] = context_cells[field]
                    continue
                context_values[field] = value
                if field in source_cells:
                    context_cells[field] = source_cells[field]

        if not schema_records:
            schema_records = self._extract_from_question_answer_layout(structure, cell_map)
        schema_records = self._prune_incomplete_procedure_rows(schema_records)
        schema_records = sorted(schema_records, key=self._record_sort_key)
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
        }

    def _resolve_excerpt_from_location(self, location, schema_records):
        sample_ids = re.findall(r"R[0-9A-Za-z-]+", location or "")
        mentioned_fields = [field for field in STANDARD_SCHEMA_FIELDS if field in (location or "")]

        candidates = []
        if sample_ids:
            for record in schema_records:
                record_sample = str(record.get("sample_id") or "")
                tokens = [token.strip() for token in record_sample.split(",") if token.strip()]
                if any(sample_id in tokens for sample_id in sample_ids):
                    candidates.append(record)
        if not candidates:
            candidates = list(schema_records)

        field_priority = mentioned_fields or [
            "test_steps",
            "audit_procedure",
            "test_result",
            "conclusion",
            "exception_flag",
            "sample_selection_method",
            "sample_size",
            "control_description",
            "control_id",
        ]

        if mentioned_fields and candidates:
            has_non_null = False
            for record in candidates:
                for field in mentioned_fields:
                    if not self._is_empty(record.get(field)):
                        has_non_null = True
                        break
                if has_non_null:
                    break
            if not has_non_null:
                return "null"

        for record in candidates:
            for field in field_priority:
                value = record.get(field)
                if self._is_empty(value):
                    continue
                snippet = re.sub(r"\s+", " ", str(value)).strip()
                if snippet:
                    return snippet[:220]
        return None

    def _excerpt_in_schema(self, excerpt, schema_text):
        if not excerpt:
            return True

        def normalize(value):
            return re.sub(r"\s+", " ", (value or "").strip())

        cleaned = normalize(excerpt.replace("原文摘录:", ""))
        cleaned = cleaned.strip("“”\"' ")
        if cleaned and cleaned in schema_text:
            return True

        quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", excerpt)
        candidates = quoted or re.split(r"[；;。|,\n]+", cleaned)
        for item in candidates:
            token = normalize(item).strip("“”\"' ")
            if len(token) >= 2 and token in schema_text:
                return True
        return False

    @staticmethod
    def _split_markdown_row(row_text):
        stripped = (row_text or "").strip()
        if not stripped.startswith("|"):
            return None

        core = stripped
        if core.startswith("|"):
            core = core[1:]
        if core.endswith("|"):
            core = core[:-1]

        cells = [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", core)]
        return cells

    @staticmethod
    def _is_markdown_separator_row(cells):
        if not cells:
            return False
        compact = [re.sub(r"\s+", "", cell or "") for cell in cells]
        return all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in compact if cell)

    @staticmethod
    def _join_markdown_row(cells):
        escaped = [str(cell).replace("|", r"\|") for cell in cells]
        return f"| {' | '.join(escaped)} |"

    def _enforce_issue_excerpt_traceability(self, review_markdown, schema_records):
        if not review_markdown or not schema_records:
            return review_markdown

        schema_chunks = []
        for record in schema_records:
            for field in STANDARD_SCHEMA_FIELDS:
                value = record.get(field)
                if not self._is_empty(value):
                    schema_chunks.append(re.sub(r"\s+", " ", str(value)).strip())
        schema_text = "\n".join(schema_chunks)
        if not schema_text:
            return review_markdown

        lines = review_markdown.splitlines()
        in_issue_table = False
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"^\|\s*问题ID\s*\|", stripped):
                in_issue_table = True
                continue
            if not in_issue_table:
                continue
            if not stripped.startswith("|"):
                if stripped:
                    in_issue_table = False
                continue

            cells = self._split_markdown_row(stripped)
            if not cells:
                continue
            if len(cells) < 7:
                continue
            if self._is_markdown_separator_row(cells):
                continue
            if len(cells) > 7:
                # Keep head/tail columns stable and fold ambiguous middle pipes
                # into the "原文摘录" column.
                cells = cells[:4] + [" | ".join(cells[4:-2])] + cells[-2:]

            location = cells[3]
            excerpt = cells[4]
            if self._excerpt_in_schema(excerpt, schema_text):
                continue

            replacement = self._resolve_excerpt_from_location(location, schema_records)
            if not replacement:
                continue

            cells[4] = replacement
            lines[idx] = self._join_markdown_row(cells)

        return "\n".join(lines)

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
            "定位必须引用字段名或 sample_id；原文摘录必须逐字来自 schema_records 中某个字段值，不可引用 schema_records 之外内容、不可编造。"
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
            raw_review = response.choices[0].message.content or ""
            return self._enforce_issue_excerpt_traceability(raw_review, payload.get("schema_records", []))
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

        columns = ["sample_id"] + STANDARD_SCHEMA_FIELDS
        table = doc.add_table(rows=1, cols=len(columns))
        table.style = "Table Grid"
        for idx, col in enumerate(columns):
            table.rows[0].cells[idx].text = col

        for record in schema_records:
            row_cells = table.add_row().cells
            for idx, col in enumerate(columns):
                value = record.get(col)
                # Keep falsy non-None values visible (0/False); only None is null.
                row_cells[idx].text = "null" if value is None else str(value)

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

            doc.add_heading("标准 Schema 抽取结果", level=3)
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
