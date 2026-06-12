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
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
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

# ---------------------------------------------------------------------------
# Severity mapping: internal P0/P1/P2 ↔ display 高/中/低
# ---------------------------------------------------------------------------
_SEVERITY_DISPLAY = {"P0": "高", "P1": "中", "P2": "低"}
_SEVERITY_FROM_CHINESE = {"高": "P0", "中": "P1", "低": "P2"}

# Maximum length for excerpt text in evidence_refs (kept consistent across scripts)
# Used in two contexts:
#   1. _repair_finding_result: constructing excerpts from snippet/basis (last-resort path)
#   2. _verify_evidence_refs: replacing mismatched excerpts with actual cell text
# 2000 chars preserves the full evidence content of a typical audit cell (~500 Chinese chars).
_EXCERPT_MAX_LEN = 2000
# When a repair-constructed excerpt comes from snippet/basis (not cell text), we
# add this marker so the auditor knows it wasn't a verbatim excerpt from the source.
_EXCERPT_CONSTRUCTED_MARKER = "[非逐字原文]"

# ---------------------------------------------------------------------------
# Unified Finding result JSON Schema — each LLM call that returns findings
# should produce objects conforming to this schema.
# ---------------------------------------------------------------------------
_FINDING_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["status", "conclusion", "evidence_refs"],
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail", "unknown"]},
        "conclusion": {"type": "string", "minLength": 4},
        "reasons": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "evidence_refs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["cell_or_range"],
                "properties": {
                    "sheet": {"type": "string"},
                    "cell_or_range": {"type": "string"},
                    "attachment": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
            },
        },
        "severity": {"type": "string", "enum": ["P0", "P1", "P2"]},
        "risk_type": {
            "type": "string",
            "enum": ["覆盖性", "一致性", "证据不足", "方法性", "逻辑性", "跨字段一致性"],
        },
        "fix_suggestion": {
            "type": "object",
            "properties": {
                "missing_field": {"type": "string"},
                "supplement_explanation": {"type": "string"},
                "required_evidence_type": {"type": "string"},
            },
        },
        "unknown_reason": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# Finding result validation & repair helpers (self-contained, not imported)
# ---------------------------------------------------------------------------

def _validate_finding_result(obj: Any, schema_records: Optional[List[dict]] = None) -> Tuple[bool, List[str]]:
    """Validate a single finding dict against _FINDING_RESULT_SCHEMA.

    Returns (valid, errors).  Beyond basic JSON Schema checks, enforces:
    - status=="fail" ⇒ evidence_refs non-empty
    - status=="unknown" ⇒ unknown_reason non-empty and ≥10 chars
    - status!="pass" ⇒ severity and risk_type required
    """
    if not isinstance(obj, dict):
        return False, ["result is not a dict"]
    errors: List[str] = []
    try:
        jsonschema.validate(obj, _FINDING_RESULT_SCHEMA)
    except jsonschema.ValidationError as exc:
        errors.append(str(exc.message))

    status = obj.get("status", "")
    # fail must have evidence_refs
    if status == "fail":
        refs = obj.get("evidence_refs") or []
        if not isinstance(refs, list) or len(refs) == 0:
            errors.append("status=fail but evidence_refs is empty")
    # unknown must have unknown_reason
    if status == "unknown":
        reason = str(obj.get("unknown_reason", "")).strip()
        if len(reason) < 10:
            errors.append("status=unknown but unknown_reason is empty or <10 chars")
    # non-pass must have severity & risk_type
    if status != "pass":
        if not obj.get("severity"):
            errors.append("status!=pass but severity is missing")
        if not obj.get("risk_type"):
            errors.append("status!=pass but risk_type is missing")
    return (len(errors) == 0, errors)


def _repair_finding_result(obj: Any, schema_records: Optional[List[dict]] = None) -> Optional[dict]:
    """Attempt to fix common issues in a finding dict.

    Returns repaired dict or None if unrepairable.
    """
    if not isinstance(obj, dict):
        return None
    repaired = dict(obj)

    # --- status migration: 旧中文值 → pass/fail/unknown ---
    old_status = str(repaired.get("status", "")).strip()
    if old_status == "无问题":
        repaired["status"] = "pass"
    elif old_status == "有问题":
        repaired["status"] = "fail"
    elif old_status == "不确定":
        repaired["status"] = "unknown"

    status = repaired.get("status", "fail")

    # --- severity migration: 高/中/低 → P0/P1/P2 ---
    sev = str(repaired.get("severity", "")).strip()
    if sev in _SEVERITY_FROM_CHINESE:
        repaired["severity"] = _SEVERITY_FROM_CHINESE[sev]
    elif sev not in ("P0", "P1", "P2", ""):
        repaired["severity"] = "P1"
    if status != "pass" and not repaired.get("severity"):
        repaired["severity"] = "P1"

    # --- conclusion: derive from basis if missing ---
    if not repaired.get("conclusion"):
        basis = str(repaired.get("basis", "")).strip()
        if basis:
            repaired["conclusion"] = basis[:200]
        else:
            repaired["conclusion"] = f"发现{status}类问题"

    # --- evidence_refs: construct from related_cells + snippet if missing ---
    refs = repaired.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        constructed: List[dict] = []
        related = repaired.get("related_cells") or repaired.get("cell") or ""
        if isinstance(related, list):
            cells = related
        elif isinstance(related, str):
            cells = [c.strip() for c in re.split(r"[,;，；\s]+", related) if c.strip()]
        else:
            cells = []
        snippet_text = str(repaired.get("snippet", "") or repaired.get("basis", "")).strip()
        for c in cells:
            ref = {"cell_or_range": c}
            if snippet_text:
                ref["excerpt"] = snippet_text[:_EXCERPT_MAX_LEN]
            constructed.append(ref)
        if not constructed and snippet_text:
            constructed.append({"cell_or_range": "", "excerpt": snippet_text[:_EXCERPT_MAX_LEN]})
        # Tag constructed excerpts so auditors know they weren't from LLM output
        for ref in constructed:
            if isinstance(ref, dict) and ref.get("cell_or_range"):
                ref["cell_or_range"] = str(ref.get("cell_or_range", "")) + _EXCERPT_CONSTRUCTED_MARKER
        repaired["evidence_refs"] = constructed

    # --- fail with empty evidence_refs ⇒ downgrade to unknown ---
    if repaired.get("status") == "fail":
        refs = repaired.get("evidence_refs") or []
        if not isinstance(refs, list) or len(refs) == 0:
            repaired["status"] = "unknown"
            repaired["unknown_reason"] = "无法引用原始证据佐证该判定，降级为不确定"
            repaired["severity"] = "P2"
            status = "unknown"

    # --- risk_type: default if missing ---
    if status != "pass" and not repaired.get("risk_type"):
        repaired["risk_type"] = "证据不足"

    # --- unknown_reason: auto-generate if missing ---
    if status == "unknown":
        reason = str(repaired.get("unknown_reason", "")).strip()
        if len(reason) < 10:
            repaired["unknown_reason"] = "LLM未说明不确定原因：需要补充更多信息以判定"

    # --- reasons: derive from basis if missing ---
    if not repaired.get("reasons"):
        basis = str(repaired.get("basis", "")).strip()
        if basis:
            repaired["reasons"] = [basis[:300]]
        else:
            repaired["reasons"] = [repaired.get("conclusion", "")]

    # --- fix_suggestion: derive from suggestion if missing ---
    if not repaired.get("fix_suggestion"):
        sug = str(repaired.get("suggestion", "")).strip()
        if sug:
            repaired["fix_suggestion"] = {"supplement_explanation": sug[:300]}
        else:
            repaired["fix_suggestion"] = {}

    return repaired


def _validate_llm_results(
    results_list: List[Any],
    schema_records: Optional[List[dict]] = None,
) -> Tuple[List[dict], bool]:
    """Validate and repair a list of finding dicts.

    Returns (valid_results, needs_retry).
    If any result is unrepairable, needs_retry=True.
    """
    valid: List[dict] = []
    needs_retry = False
    for obj in results_list:
        if not isinstance(obj, dict):
            needs_retry = True
            continue
        ok, errors = _validate_finding_result(obj, schema_records)
        if ok:
            valid.append(obj)
        else:
            repaired = _repair_finding_result(obj, schema_records)
            if repaired is not None:
                ok2, _ = _validate_finding_result(repaired, schema_records)
                if ok2:
                    valid.append(repaired)
                else:
                    needs_retry = True
            else:
                needs_retry = True
    return valid, needs_retry


def _excerpt_matches(excerpt: str, actual_text: str) -> bool:
    """Check if excerpt is a substring of actual_text after normalisation."""
    _WS_RE = re.compile(r"\s+")
    _PUNCT_RE = re.compile(r"[^\w一-鿿]+", re.UNICODE)
    norm_ex = _PUNCT_RE.sub("", _WS_RE.sub("", excerpt)).lower()
    norm_at = _PUNCT_RE.sub("", _WS_RE.sub("", actual_text)).lower()
    if not norm_ex or not norm_at:
        return False
    return norm_ex in norm_at


def _verify_evidence_refs(evidence_refs: List[dict], ws) -> List[dict]:
    """Verify evidence_refs excerpts match actual cell text; repair if possible."""
    if not ws:
        return evidence_refs
    verified: List[dict] = []
    for ref in evidence_refs:
        if not isinstance(ref, dict):
            continue
        cell = ref.get("cell_or_range", "")
        excerpt = ref.get("excerpt", "")
        actual_text = ""
        if cell:
            try:
                c = ws[cell]
                val = c.value
                actual_text = str(val).strip() if val is not None else ""
            except Exception:
                pass
        if actual_text and _excerpt_matches(excerpt, actual_text):
            verified.append(ref)
        elif actual_text:
            verified.append({**ref, "excerpt": actual_text[:220]})
    return verified


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
        """Review a sheet using structured Excel content (no image recognition).

        Returns either a dict (structured JSON path) or a Markdown string
        (legacy fallback).  Callers can detect via ``isinstance(result, dict)``.
        """
        payload = self._build_sheet_review_payload(sheet_name)

        # ---- 尝试结构化 JSON 路径 ----
        json_result = self._try_review_structured_sheet_json(sheet_name, payload)
        if json_result is not None:
            return json_result

        # ---- fallback: 旧 Markdown 路径 ----
        return self._review_structured_sheet_markdown(sheet_name, payload)

    def _try_review_structured_sheet_json(self, sheet_name, payload):
        """Request JSON output conforming to the unified schema.  Returns dict or None."""
        prompt = (
            "你是IT审计底稿审阅专家。输入数据来自 openpyxl 对 Excel 的结构化读取，不是截图。\n"
            "请按以下四个维度审阅：\n"
            "A. 覆盖性：测试过程是否覆盖审计程序要求。\n"
            "B. 方法性问题：识别如仅询问无证据、只测设计不测执行、抽样依据不清、样本量不足、无例外闭环、证据与结论不匹配。\n"
            "C. 逻辑问题（内部自洽）：测试结果、结论、步骤之间矛盾。\n"
            "D. 跨字段一致性：控制描述/审计程序/测试过程/测试结果/结论/例外标记/期间与样本日期。\n\n"
            "输出要求：必须输出严格JSON对象，格式为：\n"
            "{\n"
            "  \"overview\": {\"coverage_status\": \"完整/部分覆盖/未覆盖\", "
            "\"missing_points\": \"...\", \"risk_impact\": \"...\"},\n"
            "  \"issues\": [\n"
            "    {\n"
            "      \"status\": \"fail\",\n"
            "      \"conclusion\": \"一句话结论\",\n"
            "      \"reasons\": [\"要点1\", \"要点2\"],\n"
            "      \"evidence_refs\": [{\"sheet\": \"...\", \"cell_or_range\": \"...\", "
            "\"attachment\": \"...\", \"excerpt\": \"...\"}],\n"
            "      \"severity\": \"P0\",\n"
            "      \"risk_type\": \"覆盖性\",\n"
            "      \"fix_suggestion\": {\"missing_field\": \"...\", "
            "\"supplement_explanation\": \"...\", \"required_evidence_type\": \"...\"}\n"
            "    }\n"
            "  ],\n"
            "  \"evidence_gaps\": [\"...\"]\n"
            "}\n\n"
            "关键约束：\n"
            "1) status=fail时必须至少1个evidence_ref，excerpt必须逐字来自schema_records中某个字段值\n"
            "2) 无法引用原文时status必须为unknown，unknown_reason必填（≥10字符）\n"
            "3) severity使用P0/P1/P2；risk_type从[覆盖性/一致性/证据不足/方法性/逻辑性/跨字段一致性]选一\n"
            "4) 不要输出Markdown代码块"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是严谨的IT审计底稿审阅专家，输出严格JSON结构化审阅结果。",
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
                max_tokens=3000,
                temperature=0.2,
                timeout=180,
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = self._try_parse_review_json(raw)
            if parsed is None:
                return None
            # 验证 + 修复 issues
            issues = parsed.get("issues") or []
            valid_issues, needs_retry = _validate_llm_results(issues, payload.get("schema_records", []))
            parsed["issues"] = valid_issues
            # 对每个 issue 的 evidence_refs 做进一步验证
            for issue in parsed["issues"]:
                refs = issue.get("evidence_refs")
                if isinstance(refs, list) and refs and self._workbook is not None:
                    sheet_obj = self._workbook[sheet_name] if sheet_name in self._workbook.sheetnames else None
                    if sheet_obj is not None:
                        issue["evidence_refs"] = _verify_evidence_refs(refs, sheet_obj)
            return parsed
        except Exception:
            return None

    @staticmethod
    def _try_parse_review_json(text):
        """Robust JSON object parser; returns dict or None."""
        if not text:
            return None
        s = text.strip()
        # strip code fences if any
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```\s*$", "", s)
        start = s.find("{")
        if start < 0:
            return None
        candidate = s[start:]
        # find balanced closing brace
        depth = 0
        end = -1
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return None
        try:
            obj = json.loads(candidate[:end])
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _review_structured_sheet_markdown(self, sheet_name, payload):
        """Legacy Markdown review path — kept as fallback."""
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

    def _append_structured_review_to_doc(self, doc, review):
        """Render a structured review (dict) into the DOCX.

        Expected structure::

            {
              "overview": {"coverage_status", "missing_points", "risk_impact"},
              "issues": [
                {
                  "status", "conclusion", "reasons", "evidence_refs",
                  "severity", "risk_type", "fix_suggestion",
                  "unknown_reason"
                }
              ],
              "evidence_gaps": [...]
            }
        """
        if not isinstance(review, dict):
            self._append_markdown_to_doc(doc, str(review))
            return

        overview = review.get("overview") or {}
        if overview:
            doc.add_paragraph()
            doc.add_paragraph("【审阅总览】").runs[0].bold = True
            for k, v in overview.items():
                doc.add_paragraph(f"{k}: {v}")

        issues = review.get("issues") or []
        if issues:
            doc.add_paragraph()
            doc.add_paragraph(f"【问题清单】（共 {len(issues)} 项）").runs[0].bold = True
            table = doc.add_table(rows=1, cols=8)
            table.style = "Light Grid Accent 1"
            hdr = table.rows[0].cells
            headers = ["状态", "严重级别", "风险类型", "结论", "理由", "证据引用", "整改建议", "不确定原因"]
            for i, h in enumerate(headers):
                hdr[i].text = h
            for issue in issues:
                status = str(issue.get("status", ""))
                severity = str(issue.get("severity", ""))
                # severity → display
                severity_disp = _SEVERITY_DISPLAY.get(severity, severity)
                risk_type = str(issue.get("risk_type", ""))
                conclusion = str(issue.get("conclusion", ""))
                reasons = issue.get("reasons") or []
                if isinstance(reasons, list):
                    reasons_text = " | ".join(str(r) for r in reasons[:5])
                else:
                    reasons_text = str(reasons)
                refs = issue.get("evidence_refs") or []
                if isinstance(refs, list):
                    refs_text_parts = []
                    for r in refs[:5]:
                        if not isinstance(r, dict):
                            continue
                        cell = r.get("cell_or_range", "")
                        excerpt = (r.get("excerpt") or "")[:80]
                        attachment = r.get("attachment", "")
                        piece = cell or "(无坐标)"
                        if excerpt:
                            piece += f": {excerpt}"
                        if attachment:
                            piece += f" [附件: {attachment}]"
                        refs_text_parts.append(piece)
                    refs_text = "\n".join(refs_text_parts)
                else:
                    refs_text = ""
                fix_sug = issue.get("fix_suggestion") or {}
                if isinstance(fix_sug, dict):
                    fix_parts = []
                    if fix_sug.get("missing_field"):
                        fix_parts.append(f"缺: {fix_sug['missing_field']}")
                    if fix_sug.get("supplement_explanation"):
                        fix_parts.append(f"补: {fix_sug['supplement_explanation'][:200]}")
                    if fix_sug.get("required_evidence_type"):
                        fix_parts.append(f"需证据: {fix_sug['required_evidence_type'][:100]}")
                    fix_text = " | ".join(fix_parts)
                else:
                    fix_text = str(fix_sug)
                unknown = str(issue.get("unknown_reason", ""))
                row = table.add_row().cells
                row[0].text = status
                row[1].text = severity_disp
                row[2].text = risk_type
                row[3].text = conclusion[:300]
                row[4].text = reasons_text[:400]
                row[5].text = refs_text[:400]
                row[6].text = fix_text[:400]
                row[7].text = unknown[:200]
        else:
            doc.add_paragraph("未发现重大问题。")

        gaps = review.get("evidence_gaps") or []
        if gaps:
            doc.add_paragraph()
            doc.add_paragraph("【需补充证据】").runs[0].bold = True
            for g in gaps:
                doc.add_paragraph(f"• {g}")

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
            if isinstance(review, dict):
                self._append_structured_review_to_doc(doc, review)
            else:
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
