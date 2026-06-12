import os
import re
import argparse
import dataclasses
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jsonschema
import openpyxl
from openpyxl.utils import get_column_letter


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


@dataclass(frozen=True)
class Finding:
    issue_type: str
    severity: str  # 内部 P0/P1/P2，输出映射 高/中/低
    sheet: str
    cell: Optional[str]
    snippet: str
    basis: str
    suggestion: str
    # --- 新增字段（均有默认值，向后兼容） ---
    status: str = "fail"  # pass / fail / unknown
    risk_type: str = ""  # 覆盖性 / 一致性 / 证据不足 / ...
    evidence_refs: str = "[]"  # JSON string of list[dict]
    conclusion: str = ""
    reasons: str = "[]"  # JSON string of list[str]
    fix_suggestion_detail: str = "{}"  # JSON string of dict
    unknown_reason: str = ""
    needs_review: bool = False


@dataclass(frozen=True)
class AttachmentPreviewItem:
    index: str
    rel_dir: str
    filename: str
    rel_path: str
    file_type: str
    description: str
    status: str


# ---------------------------------------------------------------------------
# Finding result validation & repair helpers
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
    # Keep as-is if already pass/fail/unknown

    status = repaired.get("status", "fail")

    # --- severity migration: 高/中/低 → P0/P1/P2 ---
    sev = str(repaired.get("severity", "")).strip()
    if sev in _SEVERITY_FROM_CHINESE:
        repaired["severity"] = _SEVERITY_FROM_CHINESE[sev]
    elif sev not in ("P0", "P1", "P2", ""):
        repaired["severity"] = "P1"  # default to medium
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
    constructed_from_fallback = False
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
            ref: dict = {"cell_or_range": c}
            if snippet_text:
                ref["excerpt"] = snippet_text[:_EXCERPT_MAX_LEN]
            constructed.append(ref)
        if not constructed and snippet_text:
            constructed.append({"cell_or_range": "", "excerpt": snippet_text[:_EXCERPT_MAX_LEN]})
        constructed_from_fallback = bool(constructed)
        # Tag constructed excerpts so the auditor knows their origin
        if constructed_from_fallback and constructed:
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


def _get_cell_text(ws, cell_ref: str) -> str:
    """Safely get cell text from a worksheet.

    Strips any _EXCERPT_CONSTRUCTED_MARKER suffix from cell_ref before lookup.
    """
    if not cell_ref or not ws:
        return ""
    # Strip constructed marker for cell lookup
    clean_ref = cell_ref.replace(_EXCERPT_CONSTRUCTED_MARKER, "").strip()
    if not clean_ref:
        return ""
    try:
        cell = ws[clean_ref]
        val = cell.value
        return str(val).strip() if val is not None else ""
    except Exception:
        return ""


def _verify_evidence_refs(evidence_refs: List[dict], ws) -> List[dict]:
    """Verify evidence_refs excerpts match actual cell text; repair if possible.

    Constructed refs (with _EXCERPT_CONSTRUCTED_MARKER suffix) are preserved as-is
    since they were built from snippet/basis, not from real cell content.
    """
    if not ws:
        return evidence_refs
    verified: List[dict] = []
    for ref in evidence_refs:
        if not isinstance(ref, dict):
            continue
        cell = ref.get("cell_or_range", "")
        excerpt = ref.get("excerpt", "")

        # Constructed refs: keep as-is (can't verify against cell text)
        if _EXCERPT_CONSTRUCTED_MARKER in str(cell):
            verified.append(ref)
            continue

        actual_text = _get_cell_text(ws, cell)
        if actual_text and _excerpt_matches(excerpt, actual_text):
            verified.append(ref)
        elif actual_text:
            # 修复：替换为实际单元格文本
            verified.append({**ref, "excerpt": actual_text[:_EXCERPT_MAX_LEN]})
        # else: cell 无效或为空，丢弃该 evidence_ref
    return verified


# ---------------------------------------------------------------------------
# Hallucination-reduction helpers
# ---------------------------------------------------------------------------

_EXCEPTION_FLAG_TOKENS = ("是", "有异常", "Y", "异常", "缺陷", "未通过")


def _cross_validate_finding(finding: "Finding", wb) -> List[str]:
    """Deterministic cross-checks against the workbook.

    Returns a list of issue codes; empty list means no issues.
    Issue codes:
      - "exception_flag_contradicts_pass": status=pass but exception_flag cell is positive
      - "coverage_claim_but_no_sample_size": risk_type=覆盖性 but sample_size empty
      - "evidence_excerpt_mismatch": evidence_ref excerpt doesn't match cell
      - "high_severity_no_evidence": severity=P0/fail but no evidence_refs
    """
    issues: List[str] = []
    sheet = finding.sheet
    cell_refs: List[str] = []
    if finding.cell:
        for c in str(finding.cell).split(","):
            c = c.strip()
            if c:
                cell_refs.append(c)
    try:
        refs = json.loads(finding.evidence_refs) if finding.evidence_refs else []
    except Exception:
        refs = []
    if isinstance(refs, list):
        for r in refs:
            if isinstance(r, dict) and r.get("cell_or_range"):
                cell_refs.append(r["cell_or_range"])

    if not wb or sheet not in wb.sheetnames:
        return issues
    ws = wb[sheet]

    # 1) status=pass but exception_flag 单元格含异常标记
    if finding.status == "pass":
        for c in cell_refs:
            txt = _get_cell_text(ws, c)
            if txt and any(tok in txt for tok in _EXCEPTION_FLAG_TOKENS):
                issues.append("exception_flag_contradicts_pass")
                break

    # 2) risk_type=覆盖性 但 sample_size 缺失
    if finding.risk_type == "覆盖性":
        # search schema records (sheet 已知情况下简单扫描) for sample_size
        found_sample_size = False
        for row in ws.iter_rows(values_only=False, min_row=1, max_row=min(80, ws.max_row or 80)):
            for c in row:
                if not c.value:
                    continue
                cv = str(c.value)
                if any(k in cv for k in ("样本量", "样本数量", "测试期间样本")):
                    # 找该行/列的非空值
                    for r in range(c.row, min(c.row + 5, ws.max_row + 1)):
                        for cc in range(c.column, min(c.column + 6, ws.max_column + 1)):
                            v = ws.cell(row=r, column=cc).value
                            if v is not None and str(v).strip() and str(v).strip() not in ("样本量", "样本数量", "测试期间样本"):
                                found_sample_size = True
                                break
                        if found_sample_size:
                            break
                if found_sample_size:
                    break
            if found_sample_size:
                break
        if not found_sample_size:
            issues.append("coverage_claim_but_no_sample_size")

    # 3) evidence excerpt mismatch
    for r in refs if isinstance(refs, list) else []:
        if not isinstance(r, dict):
            continue
        cell = r.get("cell_or_range", "")
        excerpt = r.get("excerpt", "")
        if cell and excerpt:
            actual = _get_cell_text(ws, cell)
            if actual and not _excerpt_matches(excerpt, actual):
                issues.append("evidence_excerpt_mismatch")
                break

    # 4) P0/fail with no evidence_refs
    if finding.status == "fail" and finding.severity == "P0":
        if not refs:
            issues.append("high_severity_no_evidence")

    return issues


def _build_minimal_context(finding: "Finding", ws, max_chars: int = 2000) -> str:
    """Build a minimal context (500-2000 chars) for an LLM re-review.

    Includes:
    - cells explicitly referenced in evidence_refs
    - header row of the table region containing those cells
    - 1 row above and 1 row below
    """
    if not ws:
        return ""
    parts: List[str] = []
    # 解析 evidence_refs
    try:
        refs = json.loads(finding.evidence_refs) if finding.evidence_refs else []
    except Exception:
        refs = []
    if not isinstance(refs, list):
        refs = []
    # 找到目标 cells
    target_cells: List[str] = []
    for r in refs[:6]:
        if isinstance(r, dict) and r.get("cell_or_range"):
            target_cells.append(r["cell_or_range"])
    if not target_cells and finding.cell:
        for c in str(finding.cell).split(","):
            c = c.strip()
            if c:
                target_cells.append(c)
    if not target_cells:
        # fallback: 用 finding.snippet 中提及的 cell
        if finding.snippet:
            hits = re.findall(r"\b[A-Z]{1,3}\d{1,7}\b", finding.snippet)
            target_cells.extend(hits[:3])

    seen_rows: set = set()
    seen_cells: set = set()
    for cell in target_cells[:5]:
        actual = _get_cell_text(ws, cell)
        if actual and cell not in seen_cells:
            parts.append(f"{cell}: {actual[:160]}")
            seen_cells.add(cell)
        m = re.match(r"^([A-Z]+)(\d+)$", cell)
        if not m:
            continue
        col_letters, row_num = m.group(1), int(m.group(2))
        # 表头（行 1-3）
        for hdr_row in range(1, 4):
            key = ("hdr", hdr_row)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            for c in range(1, min(ws.max_column + 1, 12)):
                v = ws.cell(row=hdr_row, column=c).value
                if v:
                    parts.append(f"{get_column_letter(c)}{hdr_row}: {str(v)[:80]}")
        # 上下行
        for r_off in (-1, 1):
            r = row_num + r_off
            if r < 1 or r > (ws.max_row or 0):
                continue
            key = ("row", r)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            for c in range(1, min(ws.max_column + 1, 10)):
                v = ws.cell(row=r, column=c).value
                if v:
                    parts.append(f"{get_column_letter(c)}{r}: {str(v)[:120]}")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def _challenge_finding_with_llm(
    client,
    model: str,
    finding: "Finding",
    minimal_context: str,
) -> Optional[str]:
    """Run a 'challenge' LLM call to verify a P0 or needs_review finding.

    Returns "agree" / "disagree" / None on error.
    """
    if not client or not minimal_context:
        return None
    challenge_prompt = (
        "你是一名严格的审计质量复核专家，正在以质疑者的角度审阅以下复核发现。\n"
        "你的任务：判断该发现是否真实成立，或仅是表面/缺证据/逻辑不严。\n\n"
        f"【finding JSON】\n{finding.basis[:1000]}\n\n"
        f"【相关最小上下文（底稿原文片段）】\n{minimal_context[:1500]}\n\n"
        "请回答：agree（成立）/disagree（不成立/无依据）。只输出一个词。"
    )
    try:
        answer = _llm_chat(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": "你是审计复核的质疑者。"},
                {"role": "user", "content": challenge_prompt},
            ],
            stage="challenge",
            max_attempts=2,
            temperature=0.1,
            max_tokens=64,
        )
        answer = (answer or "").strip().lower()
        if "disagree" in answer or "不同意" in answer or "不成立" in answer:
            return "disagree"
        if "agree" in answer or "同意" in answer or "成立" in answer:
            return "agree"
        return None
    except Exception:
        return None


def resolve_llm_config() -> Tuple[Optional[str], Optional[str], str]:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        from dotenv import load_dotenv
    except Exception as e:
        load_dotenv = None
        _ = e
    if load_dotenv and os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=False)
    env_map: Dict[str, str] = {}
    if (not load_dotenv) and os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw or raw.startswith("#") or "=" not in raw:
                        continue
                    key, value = raw.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key:
                        env_map[key] = value
        except Exception:
            env_map = {}

    base_url = (
        env_map.get("CHECKER_API_BASE_URL")
        or env_map.get("LLM_API_BASE_URL")
        or env_map.get("API_BASE_URL")
        or env_map.get("OPENAI_BASE_URL")
        or env_map.get("BASE_URL")
        or os.getenv("CHECKER_API_BASE_URL")
        or os.getenv("LLM_API_BASE_URL")
        or os.getenv("API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    if base_url:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")].rstrip("/")

    api_key = (
        env_map.get("CHECKER_API_KEY")
        or env_map.get("LLM_API_KEY")
        or env_map.get("API_KEY")
        or env_map.get("DASHSCOPE_API_KEY")
        or env_map.get("OPENAI_API_KEY")
        or os.getenv("CHECKER_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    model = (
        env_map.get("CHECKER_MODEL")
        or env_map.get("LLM_MODEL")
        or env_map.get("MODEL")
        or env_map.get("OPENAI_MODEL")
        or os.getenv("CHECKER_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o"
    )

    return api_key.strip() if api_key else None, base_url.strip() if base_url else None, model


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _get_cell_value(ws, cell_ref: str) -> Optional[str]:
    cell = ws[cell_ref]
    value = cell.value
    if value is None and ws.merged_cells.ranges:
        for merged_range in ws.merged_cells.ranges:
            if cell.coordinate in merged_range:
                value = ws.cell(row=merged_range.min_row, column=merged_range.min_col).value
                break
    if _is_empty(value):
        return None
    return str(value).strip()


def _truncate(text: str, n: int = 160) -> str:
    s = (text or "").strip()
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _detect_layout(ws) -> Tuple[Optional[int], int, List[int]]:
    max_scan_row = min(ws.max_row or 0, 40)
    max_scan_col = min(ws.max_column or 0, 30)
    for r in range(1, max_scan_row + 1):
        standard_col = None
        exec_cols: List[int] = []
        for c in range(1, max_scan_col + 1):
            v = _get_cell_value(ws, f"{get_column_letter(c)}{r}")
            if not v:
                continue
            if ("标准" in v and "审计程序" in v) or ("标准审计程序" in v):
                standard_col = c
            if "执行" in v and "审计程序" in v:
                exec_cols.append(c)
        if standard_col and exec_cols:
            exec_cols = sorted({c for c in exec_cols if c != standard_col})
            if exec_cols:
                return r, standard_col, exec_cols
    return None, 0, []


def _extract_sheet_text_cells(ws) -> Iterable[Tuple[str, str]]:
    for row in ws.iter_rows(values_only=False):
        for cell in row:
            if _is_empty(cell.value):
                continue
            value = cell.value
            if isinstance(value, str):
                text = value.strip()
            else:
                text = str(value).strip()
            if not text:
                continue
            yield cell.coordinate, text


EVIDENCE_KEYWORDS = ("截图", "导出", "清单", "日志", "台账", "审批", "邮件", "报告", "附件", "协议", "工单", "记录")
INTERVIEW_ONLY_KEYWORDS = ("访谈", "询问", "口头", "沟通")
OS_DB_KEYWORDS = ("操作系统", "OS", "数据库", "DB", "DBA", "sa", "root")
CHECKPOINT_VOCAB = (
    "系统导出",
    "用户清单",
    "角色清单",
    "权限明细",
    "参数界面",
    "配置截图",
    "变更日志",
    "变更台账",
    "任务清单",
    "批处理",
    "定时任务",
    "作业调度",
    "运行日志",
    "告警",
    "工单",
    "审批",
    "授权",
    "协议",
    "合同",
    "操作系统",
    "数据库",
    "全量",
    "跨期比对",
    "账号创建时间",
    "变更时间",
    "末级权限",
    "权限矩阵",
)

ATTACHMENT_FILE_RE = re.compile(
    r"([0-9A-Za-z_\-\.\u4e00-\u9fff]+?\.(?:png|jpg|jpeg|pdf|xlsx|xls|docx|doc))",
    re.IGNORECASE,
)
ATTACHMENT_PATH_RE = re.compile(
    r"([0-9A-Za-z_\-\.\u4e00-\u9fff]+(?:[\\/][0-9A-Za-z_\-\.\u4e00-\u9fff]+)+\.(?:png|jpg|jpeg|pdf|xlsx|xls|docx|doc))",
    re.IGNORECASE,
)
ATTACHMENT_INDEX_RE = re.compile(r"(?:附件|证据|图片|截图|索引|目录索引)\s*([0-9]{1,3})")

LLM_CALL_STATS: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))


def _llm_stat(stage: str, key: str, n: int = 1) -> None:
    if not stage or not key:
        return
    try:
        LLM_CALL_STATS[str(stage)][str(key)] += int(n)
    except Exception:
        return


def _classify_llm_error(err: Exception) -> str:
    s = str(err or "").strip().lower()
    if not s:
        return "other"
    if "timed out" in s or "timeout" in s:
        return "timeout"
    if "rate limit" in s or "429" in s:
        return "rate_limit"
    if "context length" in s or "maximum context" in s or "max tokens" in s:
        return "context"
    if "json" in s and ("parse" in s or "nonjson" in s or "非json" in s):
        return "parse"
    if "502" in s or "503" in s or "504" in s or "bad gateway" in s or "gateway" in s:
        return "server"
    return "other"


def _llm_backoff_sleep(attempt: int, err_type: str) -> None:
    base = 1.2 * max(1, int(attempt))
    if err_type == "rate_limit":
        base = max(base, 6.0) * max(1, int(attempt))
    elif err_type in {"server", "timeout"}:
        base = max(base, 3.0) * max(1, int(attempt))
    time.sleep(min(30.0, base))


def _llm_chat(
    *,
    client,
    model: str,
    messages: List[Dict[str, str]],
    stage: str,
    max_attempts: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """Unified LLM call entry point with stats tracking, backoff, and retry.

    Returns the response content string. Raises on failure after all retries.
    """
    last_error: Optional[str] = None
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            _llm_stat(stage, "calls", 1)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            _llm_stat(stage, "ok", 1)
            return content or ""
        except Exception as e:
            last_error = str(e)
            err_type = _classify_llm_error(e)
            _llm_stat(stage, f"error_{err_type}", 1)
            if attempt < max_attempts:
                _llm_backoff_sleep(attempt, err_type)
            continue
    raise RuntimeError(last_error or "LLM调用失败")


def _llm_request_json_list(
    *,
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    stage: str,
    max_attempts: int = 3,
    result_schema: Optional[Dict[str, Any]] = None,
    schema_records: Optional[List[dict]] = None,
) -> Tuple[Optional[List[object]], Optional[str]]:
    """Unified entry for LLM calls that return JSON list responses ({results: [...]}).

    Transport errors (rate limit, timeout, server errors) are handled internally
    by _llm_chat with proper differentiated backoff. This outer loop only retries
    on JSON parse failures or schema validation failures.

    If ``result_schema`` is provided, each item in the returned list is validated
    against it (and repaired if possible).  If any item cannot be repaired, the
    user prompt is amended with the validation error and a retry is triggered.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Optional[str] = None
    last_validation_errors: List[str] = []
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            # If we have prior validation errors, augment the user prompt for retry
            current_messages = messages
            if last_validation_errors and attempt > 1:
                err_text = "；".join(last_validation_errors[:3])
                retry_note = (
                    f"\n\n[Retry hint] 上一次输出未通过结构化校验，问题：{err_text}。"
                    f"请严格按 system 字段定义重新输出，"
                    f"确保 status=pass/fail/unknown、severity=P0/P1/P2、"
                    f"fail 时 evidence_refs 必填且 excerpt 逐字来自原文。"
                )
                current_messages = [
                    messages[0],
                    {"role": "user", "content": user_prompt + retry_note},
                ]
            content = _llm_chat(
                client=client,
                model=model,
                messages=current_messages,
                stage=stage,
                max_attempts=3,
                temperature=0.1,
                max_tokens=2048,
            )
            parsed = _try_parse_json(content)
            if isinstance(parsed, dict):
                parsed = parsed.get("results") or parsed.get("data") or parsed.get("items")
            if not isinstance(parsed, list):
                raise RuntimeError("LLM返回非JSON results 数组")
            # 验证（如果提供 schema）
            if result_schema is not None:
                valid_items, needs_retry = _validate_llm_results(parsed, schema_records)
                if needs_retry:
                    # 收集错误以供 retry
                    last_validation_errors = []
                    for obj in parsed:
                        if isinstance(obj, dict):
                            ok, errs = _validate_finding_result(obj, schema_records)
                            if not ok:
                                last_validation_errors.extend(errs)
                    if not last_validation_errors:
                        last_validation_errors = ["部分结果无法通过结构化校验"]
                    if attempt < max_attempts:
                        _llm_stat(stage, "error_schema", 1)
                        time.sleep(min(8.0, 1.5 * attempt))
                        continue
                    # 达到 max_attempts 仍有问题，使用修复后的结果
                    parsed = valid_items
            return parsed, None
        except Exception as e:
            last_error = str(e)
            # Only retry JSON parse / schema failures; transport errors already exhausted
            is_parse_error = "json" in str(e).lower() or "parse" in str(e).lower() or "非JSON" in str(e)
            if is_parse_error and attempt < max_attempts:
                _llm_stat(stage, "error_parse", 1)
                time.sleep(min(8.0, 1.5 * attempt))
                continue
            if not is_parse_error:
                break
    return None, last_error or "LLM调用失败"



def _likely_interview_only(execution_text: str) -> bool:
    t = execution_text or ""
    if not t:
        return False
    has_interview = any(k in t for k in INTERVIEW_ONLY_KEYWORDS)
    has_evidence = any(k in t for k in EVIDENCE_KEYWORDS)
    return has_interview and not has_evidence


def _requires_evidence_by_standard(standard_text: str) -> Sequence[str]:
    required: List[str] = []
    t = standard_text or ""
    if any(k in t for k in ("截图", "截图", "参数", "配置")):
        required.append("截图/参数界面")
    if any(k in t for k in ("导出", "清单", "用户清单", "权限清单")):
        required.append("导出清单")
    if any(k in t for k in ("日志", "台账", "变更日志", "变更台账", "任务清单")):
        required.append("日志/台账/清单")
    if any(k in t for k in ("审批", "授权", "批准")):
        required.append("审批/授权证据")
    if any(k in t for k in ("协议", "合同", "供应商")):
        required.append("协议/合同条款")
    return required


def load_checkpoints_xlsx(checkpoints_path: str) -> Dict[str, List[str]]:
    if not checkpoints_path:
        return {}
    if not os.path.exists(checkpoints_path):
        raise FileNotFoundError(checkpoints_path)

    wb = openpyxl.load_workbook(checkpoints_path, data_only=True)
    ws = wb.active

    checkpoints_by_sheet: Dict[str, List[str]] = defaultdict(list)
    last_sheet = None
    for row in range(1, (ws.max_row or 0) + 1):
        sheet_id = ws.cell(row=row, column=1).value
        check_text = ws.cell(row=row, column=3).value
        if isinstance(sheet_id, str):
            sheet_id = sheet_id.strip()
        if isinstance(check_text, str):
            check_text = check_text.strip()

        if sheet_id:
            last_sheet = str(sheet_id).strip()
        if not last_sheet:
            continue
        if not check_text:
            continue

        checkpoints_by_sheet[last_sheet].append(str(check_text).strip())

    return dict(checkpoints_by_sheet)


def load_attachments_preview_xlsx(preview_path: str) -> Dict[str, object]:
    if not preview_path:
        return {}
    if not os.path.exists(preview_path):
        raise FileNotFoundError(preview_path)

    wb = openpyxl.load_workbook(preview_path, data_only=True)
    ws = wb["图片描述"] if "图片描述" in wb.sheetnames else wb.active

    header_row = 1
    header_map: Dict[str, int] = {}
    max_scan_col = min(ws.max_column or 0, 40)
    for c in range(1, max_scan_col + 1):
        v = ws.cell(row=header_row, column=c).value
        if not v:
            continue
        key = str(v).strip().replace("\n", "").replace(" ", "")
        if key:
            header_map[key] = c

    def _col(*names: str) -> int:
        for n in names:
            n2 = str(n).strip().replace("\n", "").replace(" ", "")
            if n2 in header_map:
                return int(header_map[n2])
        return 0

    col_index = _col("目录索引", "索引")
    col_rel_dir = _col("相对目录", "目录", "文件夹", "相对文件夹")
    col_filename = _col("附件文件名", "文件名", "附件名称", "图片文件名")
    col_rel_path = _col("相对路径", "路径", "附件路径", "图片路径")
    col_file_type = _col("文件类型", "类型", "后缀", "格式")
    col_desc = _col("详细描述", "描述", "内容", "图片描述")
    col_status = _col("状态", "校验状态", "结果")

    items: List[AttachmentPreviewItem] = []
    by_filename: Dict[str, List[AttachmentPreviewItem]] = defaultdict(list)
    by_rel_path: Dict[str, List[AttachmentPreviewItem]] = defaultdict(list)
    by_index: Dict[str, List[AttachmentPreviewItem]] = defaultdict(list)
    by_sheet_norm: Dict[str, List[AttachmentPreviewItem]] = defaultdict(list)
    status_counts: Dict[str, int] = defaultdict(int)

    sheet_tag_re = re.compile(r"\b((?:SA|PM)[-_ ]?\d{1,2}[A-Za-z]?)\b", re.IGNORECASE)
    sheet_tag_nodelim_re = re.compile(r"\b((?:SA|PM)\d{1,2}[A-Za-z]?)\b", re.IGNORECASE)

    def _extract_sheet_norms(*texts: str) -> List[str]:
        joined = " ".join(str(t or "") for t in texts if t)
        if not joined:
            return []
        norms: List[str] = []
        for m in sheet_tag_re.findall(joined):
            nm = _normalize_sheet_id(m)
            if nm and nm not in norms:
                norms.append(nm)
        for m in sheet_tag_nodelim_re.findall(joined):
            nm = _normalize_sheet_id(m)
            if nm and nm not in norms:
                norms.append(nm)
        return norms

    for r in range(2, (ws.max_row or 0) + 1):
        raw_filename = ws.cell(row=r, column=col_filename).value if col_filename else None
        raw_desc = ws.cell(row=r, column=col_desc).value if col_desc else None
        raw_rel_path = ws.cell(row=r, column=col_rel_path).value if col_rel_path else None
        raw_status = ws.cell(row=r, column=col_status).value if col_status else None
        raw_index = ws.cell(row=r, column=col_index).value if col_index else None
        raw_rel_dir = ws.cell(row=r, column=col_rel_dir).value if col_rel_dir else None
        raw_file_type = ws.cell(row=r, column=col_file_type).value if col_file_type else None

        filename = str(raw_filename).strip() if raw_filename else ""
        description = str(raw_desc).strip() if raw_desc else ""
        rel_path = str(raw_rel_path).strip() if raw_rel_path else ""
        status = str(raw_status).strip() if raw_status else ""
        index = str(raw_index).strip() if raw_index is not None else ""
        rel_dir = str(raw_rel_dir).strip() if raw_rel_dir else ""
        file_type = str(raw_file_type).strip() if raw_file_type else ""

        if not filename and not rel_path and not description:
            continue

        item = AttachmentPreviewItem(
            index=index,
            rel_dir=rel_dir,
            filename=filename,
            rel_path=rel_path,
            file_type=file_type,
            description=description,
            status=status,
        )
        items.append(item)

        if filename:
            by_filename[filename.lower()].append(item)
        if rel_path:
            by_rel_path[rel_path.lower().replace("/", "\\")].append(item)
        if index:
            by_index[index].append(item)
        for sn in _extract_sheet_norms(rel_dir, rel_path):
            by_sheet_norm[sn].append(item)
        status_counts[status or ""] += 1

    return {
        "path": preview_path,
        "items": items,
        "by_filename": dict(by_filename),
        "by_rel_path": dict(by_rel_path),
        "by_index": dict(by_index),
        "by_sheet_norm": dict(by_sheet_norm),
        "status_counts": dict(status_counts),
    }


def _extract_attachment_refs(text: str) -> Tuple[List[str], List[str], List[str]]:
    s = (text or "").strip()
    if not s:
        return [], [], []
    rel_paths = [m.group(1) for m in ATTACHMENT_PATH_RE.finditer(s)]
    filenames = [m.group(1) for m in ATTACHMENT_FILE_RE.finditer(s)]
    indices = [m.group(1) for m in ATTACHMENT_INDEX_RE.finditer(s)]
    return filenames, rel_paths, indices


def _match_preview_items(
    preview: Dict[str, object],
    *,
    filenames: Sequence[str],
    rel_paths: Sequence[str],
    indices: Sequence[str],
) -> Tuple[List[AttachmentPreviewItem], List[str]]:
    if not preview:
        return [], list(filenames)
    by_filename = preview.get("by_filename") or {}
    by_rel_path = preview.get("by_rel_path") or {}
    by_index = preview.get("by_index") or {}

    picked: List[AttachmentPreviewItem] = []
    picked_keys = set()
    missing: List[str] = []

    def _add(items: Iterable[AttachmentPreviewItem]) -> None:
        for it in items:
            key = (it.rel_path or "").lower() or (it.filename or "").lower()
            if not key:
                continue
            if key in picked_keys:
                continue
            picked_keys.add(key)
            picked.append(it)

    for idx in indices:
        lst = by_index.get(str(idx).strip())
        if isinstance(lst, list) and lst:
            _add(lst)
        else:
            missing.append(f"索引{idx}")

    for p in rel_paths:
        key = str(p).strip().lower().replace("/", "\\")
        lst = by_rel_path.get(key)
        if isinstance(lst, list) and lst:
            _add(lst)
            continue
        missing.append(p)

    for f in filenames:
        key = str(f).strip().lower()
        lst = by_filename.get(key)
        if isinstance(lst, list) and lst:
            _add(lst)
            continue
        missing.append(f)

    return picked, missing


def _compact_keywords(text: str) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", s)
    stop = {"审计", "程序", "执行", "标准", "附件", "证据", "截图", "导出", "清单", "日志", "台账"}
    out: List[str] = []
    seen = set()
    for t in tokens:
        tt = t.strip().lower()
        if not tt or tt in stop:
            continue
        if tt in seen:
            continue
        seen.add(tt)
        out.append(t.strip())
        if len(out) >= 18:
            break
    return out


def _evidence_matches_step(step_text: str, attachment_desc: str) -> bool:
    if not step_text or not attachment_desc:
        return True
    step_keys = set(_compact_keywords(step_text) + [k for k in CHECKPOINT_VOCAB if k in step_text])
    desc_keys = set(_compact_keywords(attachment_desc) + [k for k in CHECKPOINT_VOCAB if k in attachment_desc])
    if not step_keys or not desc_keys:
        return True
    inter = step_keys.intersection(desc_keys)
    return len(inter) >= 1


def _attachments_context_for_sheet(ws, preview: Dict[str, object], limit_chars: int = 6000) -> str:
    if not preview:
        return ""
    text = _build_sheet_text_for_llm(ws, max_cells=260, max_chars=24000)
    filenames, rel_paths, indices = _extract_attachment_refs(text)
    matched, _ = _match_preview_items(preview, filenames=filenames, rel_paths=rel_paths, indices=indices)
    by_sheet = preview.get("by_sheet_norm") or {}
    if isinstance(by_sheet, dict):
        norm = _normalize_sheet_id(getattr(ws, "title", "") or "")
        lst = by_sheet.get(norm)
        if isinstance(lst, list) and lst:
            seen = set(id(it) for it in matched)
            for it in lst[:80]:
                if not isinstance(it, AttachmentPreviewItem):
                    continue
                if id(it) in seen:
                    continue
                matched.append(it)
                seen.add(id(it))
                if len(matched) >= 80:
                    break
    if not matched:
        return ""
    parts: List[str] = []
    total = 0
    for it in matched[:40]:
        line = f"- {it.rel_path or it.filename} | 状态={it.status or ''} | {it.description or ''}"
        if total + len(line) + 1 > limit_chars:
            break
        parts.append(line)
        total += len(line) + 1
    return "\n".join(parts).strip()


def _normalize_sheet_id(text: str) -> str:
    s = (text or "").strip().upper()
    s = re.sub(r"[\s\-_]+", "", s)
    return s


def _split_checkpoints(text: str) -> List[str]:
    if not text:
        return []
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    parts: List[str] = []
    for chunk in normalized.split("\n"):
        s = chunk.strip()
        if not s:
            continue
        s = re.sub(r"^\s*\d+\s*[.、]\s*", "", s)
        if s:
            parts.append(s)
    if parts:
        return parts
    return [normalized.strip()]


def _extract_checkpoint_keywords(checkpoint: str) -> List[str]:
    t = checkpoint or ""
    hits = [w for w in CHECKPOINT_VOCAB if w and w in t]
    if hits:
        seen = set()
        out = []
        for h in hits:
            if h in seen:
                continue
            seen.add(h)
            out.append(h)
        return out

    segments = re.split(r"[，；。;,.、()\[\]（）]+", t)
    picked: List[str] = []
    for seg in segments:
        s = seg.strip()
        if not s:
            continue
        if len(s) < 4:
            continue
        if len(s) > 26:
            s = s[:26]
        if any("\u4e00" <= ch <= "\u9fff" for ch in s):
            picked.append(s)
        if len(picked) >= 3:
            break
    return picked or [t[:16].strip()] if t.strip() else []


def _build_sheet_text_for_llm(ws, max_cells: int = 260, max_chars: int = 24000) -> str:
    parts: List[str] = []
    total_chars = 0
    for coord, text in _extract_sheet_text_cells(ws):
        line = f"{coord}: {text}"
        if len(parts) >= max_cells:
            break
        if total_chars + len(line) + 1 > max_chars:
            break
        parts.append(line)
        total_chars += len(line) + 1
    return "\n".join(parts)


def _llm_check_sheet_by_checkpoints(
    *,
    client,
    model: str,
    ws_title: str,
    ws,
    checkpoints: Sequence[str],
    attachments_preview: Optional[Dict[str, object]] = None,
    batch_size: int = 6,
    sleep_seconds: float = 0.2,
) -> List[Finding]:
    if not checkpoints:
        return []

    flat: List[str] = []
    for raw in checkpoints:
        for item in _split_checkpoints(raw):
            if item and item.strip():
                flat.append(item.strip())

    seen = set()
    deduped: List[str] = []
    for x in flat:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)

    if not deduped:
        return []

    sheet_text = _build_sheet_text_for_llm(ws, max_cells=260, max_chars=24000)
    if not sheet_text.strip():
        return [
            Finding(
                issue_type="LLM判定：检查要点无法复核（Sheet无文本）",
                severity="P1",
                sheet=ws_title,
                cell=None,
                snippet="",
                basis="Sheet内未提取到可用于复核的文本单元格。",
                suggestion="确认该Sheet是否为图片/对象或空白；如为图片型底稿需先OCR或改用可读文本版本。",
                status="unknown",
                unknown_reason="Sheet内无文本可复核",
                risk_type="证据不足",
            )
        ]

    system_prompt = (
        "你是一名严格的IT审计/财务审计质量复核专家。\n"
        "你将收到：\n"
        "1) 某个底稿Sheet的文本化内容（每行含单元格坐标与文字）；\n"
        "2) 该Sheet对应的“检查要点”清单；\n"
        "3) （可选）该Sheet所引用的附件预览清单（附件路径/描述/状态）。\n\n"
        "你的任务：逐条检查要点，判断该Sheet是否存在相关问题（未覆盖/证据不足/表述不清/范围不全/仅访谈等）。\n"
        "重要判断规则（避免误报）：\n"
        "1) 如果Sheet内容明确写明“未执行/未开展/未进行/不存在/未对...审阅/未清查”，则这本身意味着控制未执行或存在缺陷。此时不要将“缺少过程证据”作为独立问题点重复输出；应将问题表述为“控制未执行/未开展（无清查过程证据属结果）”。\n"
        "2) 仅当Sheet声称已执行（如“已审阅/已清查/已复核/已下发确认/已收集反馈”），但未提供相应过程证据时，才输出“缺少过程证据/证据不足”。\n"
        "输出要求：必须输出严格JSON对象：{\"results\": [...]}。\n"
        "results每个元素必须包含字段：\n"
        "- id: 整数\n"
        "- checkpoint: 字符串（原检查要点）\n"
        "- status: \"pass\"/\"fail\"/\"unknown\" (无问题/有问题/不确定)\n"
        "- conclusion: 一句话结论（当status=pass时也建议给出一句话）\n"
        "- reasons: 字符串数组，2-5条要点（说明判断依据）\n"
        "- evidence_refs: 数组，每个元素含 {sheet, cell_or_range, attachment(可选), excerpt(原文摘录)}。excerpt必须逐字来自sheet_text对应单元格内容。\n"
        "  * status=fail时必须至少1个evidence_ref；无法引用原文时status必须为unknown\n"
        "  * excerpt不可编造，必须是sheet_text中能找到的原句片段\n"
        "- severity: \"P0\"/\"P1\"/\"P2\"（当status!=pass时必填）\n"
        "- risk_type: \"覆盖性\"/\"一致性\"/\"证据不足\"/\"方法性\"/\"逻辑性\"/\"跨字段一致性\"之一\n"
        "- fix_suggestion: 对象，含 {missing_field, supplement_explanation, required_evidence_type}，说明缺什么/补什么/要哪类证据\n"
        "- unknown_reason: 当status=unknown时必填，≥10字符，说明缺少什么信息\n"
        "向后兼容字段（可同时输出，但以新字段为准）：\n"
        "- basis: 字符串（可执行整改建议，保留旧版兼容）\n"
        "- suggestion: 字符串（保留旧版兼容）\n"
        "- issue_type: 字符串（保留旧版兼容）\n"
        "- related_cells: 字符串数组（保留旧版兼容，转换到evidence_refs）\n"
        "- missing_evidence: 字符串数组（保留旧版兼容，转换到fix_suggestion.required_evidence_type）\n"
        "不要输出Markdown代码块，不要输出多余文字。"
    )

    findings: List[Finding] = []
    cell_ref_re = re.compile(r"^[A-Z]{1,3}\d{1,7}$")
    for start in range(0, len(deduped), max(1, int(batch_size))):
        chunk = deduped[start : start + max(1, int(batch_size))]
        end = start + len(chunk)
        print(f"检查要点LLM进度({ws_title}): {start + 1}-{end}/{len(deduped)}", flush=True)
        attachments_text = _attachments_context_for_sheet(ws, attachments_preview or {}) if attachments_preview else ""
        payload = {
            "sheet": ws_title,
            "checkpoints": [{"id": start + i + 1, "checkpoint": cp} for i, cp in enumerate(chunk)],
            "sheet_text": sheet_text,
            "attachments_preview": attachments_text,
        }
        user_prompt = "请按要求逐条复核以下检查要点：\n" + json.dumps(payload, ensure_ascii=False, indent=2)

        def _consume_results(objs: List[object]) -> None:
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                # --- status 迁移: 中文 → 英文 ---
                raw_status = str(obj.get("status", "")).strip()
                if raw_status == "无问题":
                    status = "pass"
                elif raw_status == "有问题":
                    status = "fail"
                elif raw_status == "不确定":
                    status = "unknown"
                else:
                    status = raw_status  # 已经是 pass/fail/unknown
                if status == "pass":
                    continue
                checkpoint = str(obj.get("checkpoint", "")).strip()
                # --- severity 迁移: 中文 → P0/P1/P2 ---
                raw_sev = str(obj.get("severity", "")).strip()
                severity = _SEVERITY_FROM_CHINESE.get(raw_sev, raw_sev)
                if severity not in ("P0", "P1", "P2"):
                    severity = "P1" if status != "unknown" else "P2"
                issue_type = str(obj.get("issue_type", "")).strip() or (
                    "检查要点存在问题“ if status == ”fail“ else ”检查要点信息不足/不确定"
                )
                basis = str(obj.get("basis", "")).strip()
                suggestion = str(obj.get("suggestion", "")).strip()
                conclusion = str(obj.get("conclusion", "")).strip()
                reasons_raw = obj.get("reasons", [])
                reasons_list = [str(r).strip() for r in reasons_raw if r] if isinstance(reasons_raw, list) else []
                risk_type = str(obj.get("risk_type", "")).strip()
                unknown_reason = str(obj.get("unknown_reason", "")).strip()
                fix_suggestion_obj = obj.get("fix_suggestion") or {}
                if not isinstance(fix_suggestion_obj, dict):
                    fix_suggestion_obj = {}
                sheet_indicates_not_done = False
                sheet_signal_text = (basis or "") + "\n" + (conclusion or "") + "\n" + (sheet_text or "")
                if any(k in sheet_signal_text for k in ("未对", "未进行", "未开展", "未执行", "不存在", "未审阅", "未清查", "未复核")):
                    if any(k in checkpoint for k in ("清查全过程", "过程证据", "留痕", "反馈", "下发", "收集")) or any(
                        k in issue_type for k in ("证据", "缺失", "不足", "留痕", "反馈", "全过程")
                    ):
                        sheet_indicates_not_done = True
                related_cells = obj.get("related_cells", [])
                picked_cells: List[str] = []
                seen_cells = set()
                if isinstance(related_cells, list) and related_cells:
                    for c in related_cells:
                        cc = str(c).strip().upper()
                        if not cell_ref_re.match(cc):
                            continue
                        if cc in seen_cells:
                            continue
                        cell_text = _get_cell_value(ws, cc)
                        if not cell_text:
                            continue
                        seen_cells.add(cc)
                        picked_cells.append(cc)
                        if len(picked_cells) >= 6:
                            break
                if basis:
                    hits = re.findall(r"\b[A-Z]{1,3}\d{1,7}\b", basis.upper())
                    for h in hits:
                        hh = str(h).strip().upper()
                        if not cell_ref_re.match(hh):
                            continue
                        if hh in seen_cells:
                            continue
                        cell_text = _get_cell_value(ws, hh)
                        if not cell_text:
                            continue
                        seen_cells.add(hh)
                        picked_cells.append(hh)
                        if len(picked_cells) >= 6:
                            break
                cell = ",".join(picked_cells) if picked_cells else None

                # --- 收集 evidence_refs：优先新字段，回退 related_cells + 实际单元格文本 ---
                evidence_refs_list: List[dict] = []
                raw_refs = obj.get("evidence_refs")
                if isinstance(raw_refs, list) and raw_refs:
                    for ref in raw_refs:
                        if not isinstance(ref, dict):
                            continue
                        ev_cell = str(ref.get("cell_or_range", "")).strip()
                        ev_sheet = str(ref.get("sheet", "")).strip() or ws_title
                        ev_attachment = str(ref.get("attachment", "")).strip()
                        ev_excerpt = str(ref.get("excerpt", "")).strip()
                        if ev_cell:
                            evidence_refs_list.append({
                                "sheet": ev_sheet,
                                "cell_or_range": ev_cell,
                                "attachment": ev_attachment,
                                "excerpt": ev_excerpt,
                            })
                # 验证 excerpt 与实际单元格文本的匹配
                evidence_refs_list = _verify_evidence_refs(evidence_refs_list, ws)
                # 旧字段回退：把 picked_cells + snippet 组成 evidence_refs
                if not evidence_refs_list and picked_cells:
                    for cc in picked_cells:
                        ctext = _get_cell_value(ws, cc) or ""
                        if ctext:
                            evidence_refs_list.append({
                                "sheet": ws_title,
                                "cell_or_range": cc,
                                "excerpt": ctext[:_EXCERPT_MAX_LEN],
                            })

                # --- fail 必须有 evidence_refs，否则降级为 unknown ---
                if status == "fail" and not evidence_refs_list:
                    status = "unknown"
                    unknown_reason = unknown_reason or "无法引用原始证据佐证该判定，降级为不确定"
                    severity = "P2"

                missing_evidence = obj.get("missing_evidence", [])
                missing_text = ""
                if isinstance(missing_evidence, list) and missing_evidence:
                    missing_text = "缺失证据: " + "、".join(str(x).strip() for x in missing_evidence if str(x).strip())
                # 补全 fix_suggestion
                if not fix_suggestion_obj.get("required_evidence_type") and missing_evidence:
                    if isinstance(missing_evidence, list) and missing_evidence:
                        fix_suggestion_obj["required_evidence_type"] = "、".join(str(x).strip() for x in missing_evidence if str(x).strip())[:300]
                if not fix_suggestion_obj.get("supplement_explanation") and suggestion:
                    fix_suggestion_obj["supplement_explanation"] = suggestion[:300]

                # --- 构造分层结论：conclusion + reasons ---
                basis_parts: List[str] = []
                if checkpoint:
                    basis_parts.append("检查要点: " + checkpoint)
                if conclusion:
                    basis_parts.append("结论: " + conclusion)
                if reasons_list:
                    basis_parts.append("理由: " + " | ".join(reasons_list[:5]))
                if basis:
                    basis_parts.append("依据: " + basis)
                if missing_text:
                    basis_parts.append(missing_text)
                if evidence_refs_list:
                    refs_text = "; ".join(
                        f"{r.get('cell_or_range', '')}: {r.get('excerpt', '')[:200]}"
                        for r in evidence_refs_list[:3] if r.get('excerpt')
                    )
                    if refs_text:
                        basis_parts.append("引用: " + refs_text)
                if unknown_reason:
                    basis_parts.append("不确定原因: " + unknown_reason)
                basis2 = "\n".join(p for p in basis_parts if p).strip()

                snippet = ""
                if picked_cells:
                    parts: List[str] = []
                    for cc in picked_cells[:6]:
                        cell_text = _get_cell_value(ws, cc) or ""
                        if not cell_text:
                            continue
                        parts.append(f"{cc}: {_truncate(cell_text, 60)}")
                    if parts:
                        snippet = _truncate(" | ".join(parts), 220)
                if not snippet and evidence_refs_list:
                    parts: List[str] = []
                    for ref in evidence_refs_list[:3]:
                        ex = ref.get("excerpt", "")
                        cc = ref.get("cell_or_range", "")
                        if ex:
                            parts.append(f"{cc}: {_truncate(ex, 60)}")
                    if parts:
                        snippet = _truncate(" | ".join(parts), 220)
                if not snippet and basis2:
                    snippet = _truncate(basis2.replace("\n", " "), 220)

                if sheet_indicates_not_done and "检查要点" in (issue_type or "") and any(
                    k in (issue_type or "") for k in ("证据不足", "缺失", "全过程", "留痕")
                ):
                    issue_type = "检查要点-控制未执行/未开展（因此无过程证据）"
                    if severity not in ("P0", "P1"):
                        severity = "P1"
                    if not suggestion:
                        suggestion = (
                            "明确该控制在审计期间未执行的事实与影响；作为缺陷记录并提出整改：建立权限清查机制（导出清单-下发确认-收集反馈-例外处置-复核留痕），并补充后续期间执行记录。"
                        )

                findings.append(
                    Finding(
                        issue_type="LLM判定：" + issue_type,
                        severity=severity,
                        sheet=ws_title,
                        cell=cell,
                        snippet=snippet,
                        basis=_truncate(basis2 or "LLM判定存在问题/不确定", 3000),
                        suggestion=_truncate(
                            suggestion or "对照检查要点补充执行步骤与证据，并在底稿中保留可复核来源。",
                            1200,
                        ),
                        status=status,
                        risk_type=risk_type or ("证据不足" if status == "fail" else ""),
                        evidence_refs=json.dumps(evidence_refs_list, ensure_ascii=False),
                        conclusion=conclusion,
                        reasons=json.dumps(reasons_list, ensure_ascii=False),
                        fix_suggestion_detail=json.dumps(fix_suggestion_obj, ensure_ascii=False),
                        unknown_reason=unknown_reason,
                    )
                )

        stage = f"checkpoints:{ws_title}"
        parsed, last_error = _llm_request_json_list(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage=stage,
            max_attempts=3,
        )
        if parsed is not None:
            _consume_results(parsed)
        else:
            if len(chunk) > 1:
                _llm_stat(stage, "fallback_single", 1)
                for i, cp in enumerate(chunk):
                    payload1 = {
                        "sheet": ws_title,
                        "checkpoints": [{"id": start + i + 1, "checkpoint": cp}],
                        "sheet_text": sheet_text,
                        "attachments_preview": attachments_text,
                    }
                    user_prompt1 = "请按要求逐条复核以下检查要点：\n" + json.dumps(payload1, ensure_ascii=False, indent=2)
                    parsed1, err1 = _llm_request_json_list(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt1,
                        stage=stage,
                        max_attempts=3,
                    )
                    if parsed1 is not None:
                        _consume_results(parsed1)
                    else:
                        findings.append(
                            Finding(
                                issue_type="LLM判定：检查要点复核失败",
                                severity="P1",
                                sheet=ws_title,
                                cell=None,
                                snippet="",
                                basis=_truncate(f"检查要点: {cp}\nLLM调用失败: {err1}", 1200),
                                suggestion="检查LLM接口配置（.env）、网络连通性；必要时减少检查要点条数或缩短Sheet文本后重试。",
                                status="unknown",
                                unknown_reason="LLM调用失败，无法复核",
                                risk_type="证据不足",
                            )
                        )
            else:
                findings.append(
                    Finding(
                        issue_type="LLM判定：检查要点复核失败",
                        severity="P1",
                        sheet=ws_title,
                        cell=None,
                        snippet="",
                        basis=_truncate("检查要点: " + "；".join(chunk) + f"\nLLM调用失败: {last_error}", 1200),
                        suggestion="检查LLM接口配置（.env）、网络连通性；必要时减少检查要点条数或缩短Sheet文本后重试。",
                        status="unknown",
                        unknown_reason="LLM调用失败，无法复核",
                        risk_type="证据不足",
                    )
                )

        if sleep_seconds and sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    return findings


def _check_attachment_references(ws_title: str, ws, attachments_preview: Dict[str, object]) -> List[Finding]:
    if not attachments_preview:
        return []
    findings: List[Finding] = []

    used_any = False
    for coord, text in _extract_sheet_text_cells(ws):
        filenames, rel_paths, indices = _extract_attachment_refs(text)
        if not filenames and not rel_paths and not indices:
            continue
        used_any = True
        matched, missing = _match_preview_items(
            attachments_preview,
            filenames=filenames,
            rel_paths=rel_paths,
            indices=indices,
        )
        if missing:
            findings.append(
                Finding(
                    issue_type="附件证据引用未匹配到预览清单",
                    severity="P2",
                    sheet=ws_title,
                    cell=coord,
                    snippet=_truncate(text, 220),
                    basis=_truncate(
                        "引用: "
                        + "、".join(sorted({m for m in missing if m}))
                        + f"\n预览清单: {attachments_preview.get('path','')}",
                        1200,
                    ),
                    suggestion="核对底稿中的附件编号/文件名/路径是否与附件清单一致；如存在重命名或遗漏，请补齐清单或更新引用。",
                )
            )
        bad = [it for it in matched if (it.status or "").strip() and str(it.status).strip().upper() != "OK"]
        for it in bad[:5]:
            findings.append(
                Finding(
                    issue_type="附件预览状态异常",
                    severity="P1",
                    sheet=ws_title,
                    cell=coord,
                    snippet=_truncate(text, 220),
                    basis=_truncate(
                        f"附件: {it.rel_path or it.filename}\n状态: {it.status}\n描述: {it.description}",
                        1200,
                    ),
                    suggestion="检查该附件是否OCR/解析失败或内容不清晰；必要时补充可复核版本（导出清单/日志/原始报表）并在底稿中明确指向。",
                )
            )

    if not used_any:
        return []
    return findings


def _llm_check_evidence_vs_steps(
    *,
    client,
    model: str,
    ws_title: str,
    ws,
    attachments_preview: Dict[str, object],
    batch_size: int = 6,
    sleep_seconds: float = 0.2,
) -> List[Finding]:
    if not attachments_preview:
        return []
    findings: List[Finding] = []
    header_row, standard_col, execution_cols = _detect_layout(ws)
    if header_row is None or standard_col <= 0 or not execution_cols:
        return findings
    start_row = max(5, (header_row or 1) + 2)

    cases: List[Dict[str, object]] = []
    next_id = 1
    for row in range(start_row, (ws.max_row or 0) + 1):
        a_cell = f"{get_column_letter(standard_col)}{row}"
        a_text = _get_cell_value(ws, a_cell)
        if not a_text:
            continue
        for c in execution_cols:
            c_cell = f"{get_column_letter(c)}{row}"
            c_text = _get_cell_value(ws, c_cell)
            if not c_text:
                continue
            filenames, rel_paths, indices = _extract_attachment_refs(c_text)
            if not filenames and not rel_paths and not indices and not any(k in c_text for k in EVIDENCE_KEYWORDS):
                continue
            matched, missing = _match_preview_items(
                attachments_preview,
                filenames=filenames,
                rel_paths=rel_paths,
                indices=indices,
            )
            if (
                not matched
                and not filenames
                and not rel_paths
                and not indices
                and any(k in c_text for k in EVIDENCE_KEYWORDS)
                and isinstance(attachments_preview.get("by_sheet_norm"), dict)
            ):
                pool = attachments_preview.get("by_sheet_norm", {}).get(_normalize_sheet_id(ws_title))
                if isinstance(pool, list) and pool:
                    matched = [it for it in pool if isinstance(it, AttachmentPreviewItem)][:10]
            if missing and (filenames or rel_paths or indices):
                findings.append(
                    Finding(
                        issue_type="附件证据编号/文件未匹配（可能引用错误）",
                        severity="P1",
                        sheet=ws_title,
                        cell=c_cell,
                        snippet=_truncate(c_text, 220),
                        basis=_truncate(
                            "标准程序: "
                            + _truncate(a_text, 160)
                            + "\n引用未匹配: "
                            + "、".join(sorted({m for m in missing if m})),
                            1200,
                        ),
                        suggestion="核对附件编号/命名/路径与证据清单是否一致；必要时在底稿中给出可复核的相对路径或文件名，并补充证据来源说明。",
                    )
                )
            if matched:
                evidences: List[Dict[str, str]] = []
                for it in matched[:10]:
                    evidences.append(
                        {
                            "path": str(it.rel_path or it.filename or ""),
                            "status": str(it.status or ""),
                            "description": _truncate(str(it.description or "").replace("\r\n", "\n").replace("\r", "\n"), 1200),
                        }
                    )
                cases.append(
                    {
                        "id": next_id,
                        "sheet": ws_title,
                        "row": row,
                        "standard_cell": a_cell,
                        "execution_cell": c_cell,
                        "standard_text": a_text,
                        "execution_text": c_text,
                        "evidences": evidences,
                    }
                )
                next_id += 1

    if not cases:
        return findings

    max_items = 0
    try:
        max_items = int(os.getenv("LLM_EVIDENCE_STEPS_MAX_ITEMS", "0") or 0)
    except Exception:
        max_items = 0
    if max_items and max_items > 0 and len(cases) > max_items:
        def _sample_evenly(seq: List[Dict[str, object]], k: int) -> List[Dict[str, object]]:
            if k <= 0 or len(seq) <= k:
                return list(seq)
            step = float(len(seq)) / float(k)
            picked: List[Dict[str, object]] = []
            used = set()
            for i in range(k):
                idx = int(i * step)
                if idx < 0:
                    idx = 0
                if idx >= len(seq):
                    idx = len(seq) - 1
                if idx in used:
                    continue
                used.add(idx)
                picked.append(seq[idx])
            while len(picked) < k and len(picked) < len(seq):
                idx = len(picked)
                if idx in used or idx >= len(seq):
                    break
                used.add(idx)
                picked.append(seq[idx])
            return picked

        _llm_stat(f"evidence_steps:{ws_title}", "sampled", 1)
        original = len(cases)
        cases = _sample_evenly(cases, max_items)
        findings.append(
            Finding(
                issue_type="证据-步骤一致性抽样复核（为控制LLM调用规模）",
                severity="P2",
                sheet=ws_title,
                cell=None,
                snippet="",
                basis=_truncate(f"原匹配记录数: {original}；本次LLM抽样复核: {len(cases)}（可通过环境变量LLM_EVIDENCE_STEPS_MAX_ITEMS调整）", 1200),
                suggestion="如需全量复核，请增大LLM_EVIDENCE_STEPS_MAX_ITEMS，或先缩小Sheet范围（-s）再运行。",
            )
        )

    system_prompt = (
        "你是一名严格的IT审计/财务审计质量复核专家。\n"
        "你将收到多条“标准审计程序/执行审计程序”记录，以及执行中引用的附件证据预览信息（附件路径/状态/内容描述）。\n"
        "你的任务：判断证据是否与审计步骤匹配、是否足以支持执行描述与测试结论（如“已验证/无异常/符合”等）。\n"
        "重点关注：\n"
        "1) 证据是否能证明该步骤要验证的控制点/属性（而非无关截图）。\n"
        "2) 执行描述是否仅列附件但未说明核查点/结论依据。\n"
        "3) 证据状态异常/描述模糊时，提出需要补充的证据类型与下一步动作。\n"
        "输出要求：必须输出严格JSON对象：{\"results\": [...]}。\n"
        "results中每个元素对应输入id，且必须包含字段：\n"
        "- id: 整数\n"
        "- status: \"pass\"/\"fail\"/\"unknown\" (无问题/有问题/不确定)\n"
        "- conclusion: 一句话结论\n"
        "- reasons: 字符串数组，2-5条要点\n"
        "- evidence_refs: 数组，每个元素含 {sheet, cell_or_range, attachment(可选), excerpt(原文摘录)}。excerpt必须逐字来自执行/标准/附件描述。\n"
        "  * status=fail时必须至少1个evidence_ref；无法引用时status必须为unknown\n"
        "- severity: \"P0\"/\"P1\"/\"P2\"（当status!=pass时必填）\n"
        "- risk_type: \"覆盖性\"/\"一致性\"/\"证据不足\"/\"方法性\"/\"逻辑性\"/\"跨字段一致性\"之一\n"
        "- fix_suggestion: 对象，含 {missing_field, supplement_explanation, required_evidence_type}\n"
        "- unknown_reason: 当status=unknown时必填，≥10字符\n"
        "向后兼容字段（可同时输出）：basis, suggestion, issue_type, missing_evidence\n"
        "不要输出Markdown代码块，不要输出多余文字。"
    )

    id_to_case: Dict[int, Dict[str, object]] = {int(c.get("id")): c for c in cases if isinstance(c.get("id"), int)}
    for start in range(0, len(cases), max(1, int(batch_size))):
        chunk = cases[start : start + max(1, int(batch_size))]
        end = start + len(chunk)
        print(f"证据-步骤一致性LLM进度({ws_title}): {start + 1}-{end}/{len(cases)}", flush=True)

        payload = {"sheet": ws_title, "items": chunk}
        user_prompt = "请按要求逐条复核以下记录：\n" + json.dumps(payload, ensure_ascii=False, indent=2)

        def _consume_results(objs: List[object]) -> None:
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                rid = obj.get("id")
                if not isinstance(rid, int):
                    try:
                        rid = int(str(rid).strip())
                    except Exception:
                        continue
                raw_status = str(obj.get("status", "")).strip()
                if raw_status == "无问题":
                    status = "pass"
                elif raw_status == "有问题":
                    status = "fail"
                elif raw_status == "不确定":
                    status = "unknown"
                else:
                    status = raw_status
                if status == "pass":
                    continue
                case = id_to_case.get(int(rid))
                if not case:
                    continue
                c_cell = str(case.get("execution_cell") or "").strip() or None
                c_text = str(case.get("execution_text") or "")
                a_text = str(case.get("standard_text") or "")
                raw_sev = str(obj.get("severity", "")).strip()
                severity = _SEVERITY_FROM_CHINESE.get(raw_sev, raw_sev)
                if severity not in ("P0", "P1", "P2"):
                    severity = "P1" if status != "unknown" else "P2"
                issue_type = str(obj.get("issue_type", "")).strip() or (
                    "证据与审计步骤不匹配“ if status == ”fail“ else ”证据与审计步骤信息不足/不确定"
                )
                basis = str(obj.get("basis", "")).strip()
                suggestion = str(obj.get("suggestion", "")).strip()
                conclusion = str(obj.get("conclusion", "")).strip()
                reasons_raw = obj.get("reasons", [])
                reasons_list = [str(r).strip() for r in reasons_raw if r] if isinstance(reasons_raw, list) else []
                risk_type = str(obj.get("risk_type", "")).strip()
                unknown_reason = str(obj.get("unknown_reason", "")).strip()
                fix_suggestion_obj = obj.get("fix_suggestion") or {}
                if not isinstance(fix_suggestion_obj, dict):
                    fix_suggestion_obj = {}
                missing_evidence = obj.get("missing_evidence", [])
                missing_text = ""
                if isinstance(missing_evidence, list) and missing_evidence:
                    missing_text = "缺失证据: " + "、".join(str(x).strip() for x in missing_evidence if str(x).strip())

                # 收集 evidence_refs
                evidence_refs_list: List[dict] = []
                raw_refs = obj.get("evidence_refs")
                if isinstance(raw_refs, list) and raw_refs:
                    for ref in raw_refs:
                        if not isinstance(ref, dict):
                            continue
                        ev_cell = str(ref.get("cell_or_range", "")).strip()
                        ev_sheet = str(ref.get("sheet", "")).strip() or ws_title
                        ev_attachment = str(ref.get("attachment", "")).strip()
                        ev_excerpt = str(ref.get("excerpt", "")).strip()
                        if ev_cell or ev_excerpt:
                            evidence_refs_list.append({
                                "sheet": ev_sheet,
                                "cell_or_range": ev_cell,
                                "attachment": ev_attachment,
                                "excerpt": ev_excerpt,
                            })
                # 旧字段回退
                if not evidence_refs_list and c_cell:
                    evidence_refs_list.append({
                        "sheet": ws_title,
                        "cell_or_range": c_cell,
                        "excerpt": c_text[:_EXCERPT_MAX_LEN] if c_text else "",
                    })
                # 验证
                evidence_refs_list = _verify_evidence_refs(evidence_refs_list, ws)
                # fail 必须有 evidence_refs
                if status == "fail" and not evidence_refs_list:
                    status = "unknown"
                    unknown_reason = unknown_reason or "无法引用原始证据佐证该判定，降级为不确定"
                    severity = "P2"

                if not fix_suggestion_obj.get("required_evidence_type") and missing_evidence:
                    if isinstance(missing_evidence, list) and missing_evidence:
                        fix_suggestion_obj["required_evidence_type"] = "、".join(str(x).strip() for x in missing_evidence if str(x).strip())[:300]
                if not fix_suggestion_obj.get("supplement_explanation") and suggestion:
                    fix_suggestion_obj["supplement_explanation"] = suggestion[:300]

                basis_parts: List[str] = []
                if a_text:
                    basis_parts.append("标准程序: " + _truncate(a_text, 260))
                if conclusion:
                    basis_parts.append("结论: " + conclusion)
                if reasons_list:
                    basis_parts.append("理由: " + " | ".join(reasons_list[:5]))
                if basis:
                    basis_parts.append("LLM依据: " + basis)
                if missing_text:
                    basis_parts.append(missing_text)
                if evidence_refs_list:
                    refs_text = "; ".join(
                        f"{r.get('cell_or_range', '')}: {r.get('excerpt', '')[:200]}"
                        for r in evidence_refs_list[:3] if r.get('excerpt')
                    )
                    if refs_text:
                        basis_parts.append("引用: " + refs_text)
                if unknown_reason:
                    basis_parts.append("不确定原因: " + unknown_reason)
                final_basis = "\n".join(p for p in basis_parts if p).strip()

                findings.append(
                    Finding(
                        issue_type="LLM判定：证据-步骤一致性-" + issue_type,
                        severity=severity,
                        sheet=ws_title,
                        cell=c_cell,
                        snippet=_truncate(c_text, 220),
                        basis=_truncate(final_basis or "LLM判定存在问题/不确定", 3000),
                        suggestion=_truncate(
                            suggestion
                            or "补充与该审计步骤直接对应的截图/导出清单/日志/审批等证据，并在底稿中写明「证据→核查点→结论」的对应关系。",
                            1200,
                        ),
                        status=status,
                        risk_type=risk_type or ("证据不足" if status == "fail" else ""),
                        evidence_refs=json.dumps(evidence_refs_list, ensure_ascii=False),
                        conclusion=conclusion,
                        reasons=json.dumps(reasons_list, ensure_ascii=False),
                        fix_suggestion_detail=json.dumps(fix_suggestion_obj, ensure_ascii=False),
                        unknown_reason=unknown_reason,
                    )
                )

        stage = f"evidence_steps:{ws_title}"
        parsed, last_error = _llm_request_json_list(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage=stage,
            max_attempts=3,
        )
        if parsed is not None:
            _consume_results(parsed)
        else:
            if len(chunk) > 1:
                _llm_stat(stage, "fallback_single", 1)
                for item in chunk:
                    payload1 = {"sheet": ws_title, "items": [item]}
                    user_prompt1 = "请按要求逐条复核以下记录：\n" + json.dumps(payload1, ensure_ascii=False, indent=2)
                    parsed1, err1 = _llm_request_json_list(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt1,
                        stage=stage,
                        max_attempts=3,
                    )
                    if parsed1 is not None:
                        _consume_results(parsed1)
                    else:
                        findings.append(
                            Finding(
                                issue_type="LLM判定：证据-步骤一致性复核失败",
                                severity="P1",
                                sheet=ws_title,
                                cell=None,
                                snippet="",
                                basis=_truncate(f"LLM调用失败: {err1}", 1200),
                                suggestion="检查LLM接口配置（.env）、网络连通性；必要时减少检查行数或缩短证据描述后重试。",
                            )
                        )
            else:
                findings.append(
                    Finding(
                        issue_type="LLM判定：证据-步骤一致性复核失败",
                        severity="P1",
                        sheet=ws_title,
                        cell=None,
                        snippet="",
                        basis=_truncate(f"LLM调用失败: {last_error}", 1200),
                        suggestion="检查LLM接口配置（.env）、网络连通性；必要时减少检查行数或缩短证据描述后重试。",
                    )
                )

        if sleep_seconds and sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    return findings


def _check_procedure_pairs(ws_title: str, ws) -> List[Finding]:
    findings: List[Finding] = []
    header_row, standard_col, execution_cols = _detect_layout(ws)
    if header_row is None or standard_col <= 0 or not execution_cols:
        return findings
    start_row = max(5, (header_row or 1) + 2)

    skip_a_exact = {"n/a", "na", "不适用", "有效", "无"}
    skip_a_symbol = {"√", "×", "✓", "✗"}
    skip_c_exact = {"n/a", "na", "不适用"}
    keywords_like_procedure = ("询问", "访谈", "检查", "获取", "抽取", "观察", "复核", "比对", "重新执行", "分析", "确认", "审查")
    is_design_section = False
    if header_row and standard_col > 0:
        header_text = _get_cell_value(ws, f"{get_column_letter(standard_col)}{header_row}") or ""
        is_design_section = "设计有效性" in header_text

    empty_streak = 0
    for row in range(start_row, (ws.max_row or 0) + 1):
        row_marker = _get_cell_value(ws, f"A{row}")
        if row_marker:
            m = row_marker.strip()
            if (m.startswith("*") and m[1:].isdigit()) or (m.startswith("#") and m[1:].isdigit()):
                continue
        a_text = _get_cell_value(ws, f"{get_column_letter(standard_col)}{row}")
        has_exec = any(_get_cell_value(ws, f"{get_column_letter(c)}{row}") for c in execution_cols)
        if a_text is None and not has_exec:
            empty_streak += 1
            if empty_streak >= 30:
                break
            continue
        empty_streak = 0

        if a_text is None:
            continue

        a_compact = a_text.replace(" ", "").replace("\n", "").strip()
        if a_compact in skip_a_symbol or a_compact.lower() in skip_a_exact:
            continue
        if len(a_compact) <= 10 and all(ch.isalnum() or ch in "-_./" for ch in a_compact):
            continue
        if not any(k in a_text for k in keywords_like_procedure) and "。" not in a_text and "•" not in a_text:
            continue

        required_evidence = _requires_evidence_by_standard(a_text)
        for exec_col in execution_cols:
            c_cell = f"{get_column_letter(exec_col)}{row}"
            c_text = _get_cell_value(ws, c_cell)
            if c_text is None:
                continue

            row_marker = _get_cell_value(ws, f"A{row}")
            if row_marker:
                m = row_marker.strip()
                if (m.startswith("*") and m[1:].isdigit()) or (m.startswith("#") and m[1:].isdigit()):
                    continue

            c_compact = c_text.replace(" ", "").replace("\n", "").strip()
            if c_compact.lower() in skip_c_exact:
                continue
            if len(c_compact) <= 12 and all(ch.isalnum() or ch in "-_./" for ch in c_compact):
                continue
            if len(a_text) < 20 or len(c_text) < 20:
                continue

            execution_like = (
                ("我们" in c_text)
                or any(k in c_text for k in INTERVIEW_ONLY_KEYWORDS)
                or any(k in c_text for k in EVIDENCE_KEYWORDS)
            )
            if not execution_like:
                template_like = ("•" in c_text or "被认为" in c_text or c_text.strip().startswith(("如果", "以下", "在以下", "根据", "用户参与", "文档包括")))
                if template_like:
                    findings.append(
                        Finding(
                            issue_type="执行列疑似未替换模板/未按要求填列",
                            severity="P0",
                            sheet=ws_title,
                            cell=c_cell,
                            snippet=_truncate(c_text, 220),
                            basis="执行列内容更像标准模板/判定口径（如「如果/以下/被认为」及大量条款），缺少「我们获取/检查/抽样/复核」等实际执行描述。",
                            suggestion="将该单元格补充为实际执行步骤与获取证据描述（含样本框定方法、样本来源/编号、证据链接/截图/导出）。",
                        )
                    )
                continue

            if _likely_interview_only(c_text):
                findings.append(
                    Finding(
                        issue_type="程序执行不到位/仅依赖访谈",
                        severity="P1",
                        sheet=ws_title,
                        cell=c_cell,
                        snippet=_truncate(c_text, 220),
                        basis="执行描述出现“访谈/询问/了解”等，但未体现截图、导出清单、日志、台账、审批等实质性证据。",
                        suggestion="补充系统截图/导出清单/日志台账/审批记录等，并在执行程序中明确证据来源与核查步骤。",
                    )
                )

            if required_evidence and not any(k in c_text for k in EVIDENCE_KEYWORDS):
                findings.append(
                    Finding(
                        issue_type="证据类型缺失",
                        severity="P1",
                        sheet=ws_title,
                        cell=c_cell,
                        snippet=_truncate(c_text, 220),
                        basis=f"标准审计程序要求获取/检查证据（{', '.join(required_evidence)}），但执行描述未体现对应证据。",
                        suggestion="对照标准程序逐条补齐证据（截图/清单/日志/审批/协议等），并在底稿中保留可复核的原始文件或路径。",
                    )
                )

            if any(k in a_text for k in ("账号新增", "新增账号", "开通")):
                if "入职" in c_text and not any(k in c_text for k in ("账号创建", "创建时间", "创建日期", "用户清单", "跨期比对")):
                    findings.append(
                        Finding(
                            issue_type="账号新增样本总量基准可能有误",
                            severity="P1",
                            sheet=ws_title,
                            cell=c_cell,
                            snippet=_truncate(c_text, 220),
                            basis="常见问题：样本总量应优先以用户清单“账号创建时间”为基准；仅以入职名单抽样可能遗漏外包/延期开户等。",
                            suggestion="优先获取系统用户清单含账号创建时间字段进行抽样；无该字段时，采用用户清单跨期比对+入职名单交叉验证组合确定样本总量。",
                        )
                    )

            if any(k in a_text for k in ("离职", "禁用", "删除")):
                if "已禁用" in c_text and ("抽样" in c_text or "样本" in c_text) and not any(k in c_text for k in ("离职名单", "全量", "关联", "匹配", "用户清单")):
                    findings.append(
                        Finding(
                            issue_type="离职账号禁用检查方法可能有误",
                            severity="P1",
                            sheet=ws_title,
                            cell=c_cell,
                            snippet=_truncate(c_text, 220),
                            basis="常见问题：不应从“已禁用账户”反向抽样，应将离职名单与用户清单全量关联核查账号状态与禁用时间。",
                            suggestion="获取审计期间离职名单，与系统用户清单关联，全量核查账号状态/禁用时间，必要时补充禁用工单或审批证据。",
                        )
                    )

            if "调岗" in a_text or "岗位变动" in a_text:
                if not any(k in c_text for k in ("权限变更", "权限调整", "角色调整", "禁用", "撤销", "变更")):
                    findings.append(
                        Finding(
                            issue_type="未覆盖调岗权限变更/禁用测试",
                            severity="P1",
                            sheet=ws_title,
                            cell=c_cell,
                            snippet=_truncate(c_text, 220),
                            basis="常见问题：调岗应纳入权限变更/禁用控制测试，仅看账号状态不足以覆盖权限调整实质性测试。",
                            suggestion="获取调岗人员名单与用户清单关联，框定调岗且持有账号人员范围，按调岗前后岗位权限差异抽样核查权限变更/禁用证据。",
                        )
                    )

            if any(k in a_text for k in ("密码策略", "密码", "复杂度", "锁定", "时效")):
                if is_design_section:
                    if not any(k in c_text for k in ("制度", "规程", "政策", "流程", "规定", "办法", "指引", "《", "<")):
                        findings.append(
                            Finding(
                                issue_type="设计有效性证据不足（密码策略）",
                                severity="P2",
                                sheet=ws_title,
                                cell=c_cell,
                                snippet=_truncate(c_text, 220),
                                basis="设计有效性测试通常以制度/流程/政策文件作为证据；当前执行描述未体现已获取/引用相关文件。",
                                suggestion="补充引用密码策略相关制度/规程/政策文件（文件名、条款、编号/链接），并在底稿中说明其适用系统与覆盖范围。",
                            )
                        )
                elif not any(k in c_text for k in ("截图", "参数", "配置", "界面")):
                    findings.append(
                        Finding(
                            issue_type="密码策略证据有效性不足",
                            severity="P1",
                            sheet=ws_title,
                            cell=c_cell,
                            snippet=_truncate(c_text, 220),
                            basis="常见问题：仅文字说明不足以支撑密码策略；应留存参数界面/配置截图等实质性证据。",
                            suggestion="补充密码策略参数界面截图（复杂度/锁定/有效期/历史密码等），并核对底稿描述与系统配置一致。",
                        )
                    )

            if any(k in a_text for k in ("批处理", "定时任务", "任务计划", "job", "Job")):
                if _likely_interview_only(c_text) or not any(k in c_text for k in ("任务", "日志", "清单", "导出")):
                    findings.append(
                        Finding(
                            issue_type="批处理作业证据不足/范围可能未覆盖",
                            severity="P1",
                            sheet=ws_title,
                            cell=c_cell,
                            snippet=_truncate(c_text, 220),
                            basis="常见问题：仅访谈了解批处理设置不充分；应获取任务清单/日志，并覆盖应用层、操作系统、数据库层面。",
                            suggestion="补充应用/OS/DB层面任务清单与执行日志导出，并明确是否覆盖全部相关批处理作业。",
                        )
                    )

            if any(k in a_text for k in ("变更", "发布", "上线", "迁移")):
                if _likely_interview_only(c_text) or not any(k in c_text for k in ("变更台账", "变更日志", "台账", "日志", "工单")):
                    findings.append(
                        Finding(
                            issue_type="系统变更证据不足/样本框定可能有误",
                            severity="P1",
                            sheet=ws_title,
                            cell=c_cell,
                            snippet=_truncate(c_text, 220),
                            basis="常见问题：应以变更日志/变更台账为样本总量基准抽样审批；仅依赖访谈或从审批流程反向框定无法验证“变更均经审批”。",
                            suggestion="获取变更日志/台账作为总体，按期间抽样追溯审批与测试/上线证据；补充工单、发布记录、回滚记录等。",
                        )
                    )

    return findings


def _check_sheet_scope(ws_title: str, ws) -> List[Finding]:
    findings: List[Finding] = []
    sheet_text = " ".join(text for _, text in _extract_sheet_text_cells(ws))

    if ws_title in {"SA-4c"}:
        if ("管理员" in sheet_text or "特权" in sheet_text) and not any(k in sheet_text for k in OS_DB_KEYWORDS):
            findings.append(
                Finding(
                    issue_type="特权账号识别范围可能不完整",
                    severity="P1",
                    sheet=ws_title,
                    cell=None,
                    snippet=_truncate(sheet_text, 220),
                    basis="常见问题：项目检查范围未覆盖操作系统与数据库层面的管理员账号设置情况，可能导致特权账号识别不完整。",
                    suggestion="补充获取并核对OS/DB层面管理员账号清单（或截图/导出），并评估与应用层管理员职责冲突与共享风险。",
                )
            )

    if ws_title in {"PM-5", "PM-6", "PM-4b", "PM-4c", "PM-4e", "PM-3"}:
        if "供应商" in sheet_text and not any(k in sheet_text for k in ("协议", "合同", "SaaS", "托管", "权利义务")):
            findings.append(
                Finding(
                    issue_type="供应商托管场景证据可能不足",
                    severity="P1",
                    sheet=ws_title,
                    cell=None,
                    snippet=_truncate(sheet_text, 220),
                    basis="常见问题：供应商维护/托管时需获取相关协议文件检查双方权利义务与实际履行情况。",
                    suggestion="补充协议/合同/运维报告/工单等证据，明确管理员账号归属（租户级权限）与监控复核责任。",
                )
            )

    return findings


def _extract_actor_candidates(ws) -> Dict[str, List[Tuple[str, str]]]:
    admin_pattern = re.compile(
        r"(?:管理员账号|系统管理员|超级管理员|admin账号|管理员用户|管理员)\s*(?:为|是|：|:)?\s*([A-Za-z0-9_\-]{2,20}|[\u4e00-\u9fff]{2,4})"
    )
    executor_pattern = re.compile(
        r"(?:执行人|操作人|申请人|审批人|复核人)\s*(?:为|是|：|:)?\s*([A-Za-z0-9_\-]{2,20}|[\u4e00-\u9fff]{2,4})"
    )

    admins: List[Tuple[str, str]] = []
    executors: List[Tuple[str, str]] = []
    token_stopwords = {
        "职位如下",
        "调查表",
        "账号",
        "都是经过",
        "手动记录",
        "进行测试",
        "负责变更",
        "系统管理",
        "适当的个",
        "适当的人",
        "相关最终",
        "授权用户",
        "开通权限",
        "执行修改",
    }
    for coord, text in _extract_sheet_text_cells(ws):
        for m in admin_pattern.finditer(text):
            token = m.group(1)
            if token in token_stopwords or any(x in token for x in ("岗位", "职责", "权限", "人员", "管理")):
                continue
            admins.append((coord, token))
        for m in executor_pattern.finditer(text):
            token = m.group(1)
            if token in token_stopwords or any(x in token for x in ("岗位", "职责", "权限", "人员", "管理", "执行")):
                continue
            executors.append((coord, token))
    return {"admins": admins, "executors": executors}


def _parse_sheet_filter(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.lower() in {"all", "*", "全部"}:
        return None
    parts = [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]
    if not parts:
        return None
    seen = set()
    out: List[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out or None


def _extract_context_cells(ws, keywords: Sequence[str], limit: int = 8) -> List[Tuple[str, str]]:
    if not keywords:
        return []
    hits: List[Tuple[str, str]] = []
    for coord, text in _extract_sheet_text_cells(ws):
        if any(k and k in text for k in keywords):
            hits.append((coord, _truncate(text, 220)))
            if len(hits) >= limit:
                break
    return hits


def _ensure_openai_client(api_key: str, base_url: Optional[str]):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"缺少依赖 openai 或导入失败: {e}")
    timeout_raw = (
        os.getenv("CHECKER_TIMEOUT")
        or os.getenv("LLM_TIMEOUT")
        or os.getenv("OPENAI_TIMEOUT")
        or os.getenv("API_TIMEOUT")
        or ""
    )
    timeout_s = 180.0
    if str(timeout_raw).strip():
        try:
            timeout_s = float(str(timeout_raw).strip())
        except Exception:
            timeout_s = 180.0

    kwargs = {"api_key": api_key, "timeout": timeout_s, "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url

    try:
        import httpx
        from openai import DefaultHttpxClient

        http_timeout = httpx.Timeout(timeout_s, connect=min(10.0, timeout_s), read=timeout_s, write=timeout_s, pool=min(10.0, timeout_s))
        kwargs["http_client"] = DefaultHttpxClient(timeout=http_timeout)
    except Exception:
        pass

    return OpenAI(**kwargs)


def _try_parse_json(text: str):
    if not text:
        return None
    s = str(text).strip()
    start = None
    for i, ch in enumerate(s):
        if ch in "[{":
            start = i
            break
    if start is None:
        return None
    candidate = s[start:]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _llm_review_findings(
    wb,
    findings_sorted: Sequence[Finding],
    model: str,
    api_key: str,
    base_url: Optional[str],
    batch_size: int,
    sleep_seconds: float,
) -> Dict[int, Dict[str, str]]:
    client = _ensure_openai_client(api_key=api_key, base_url=base_url)

    selected: List[Tuple[int, Finding]] = []
    for idx, item in enumerate(findings_sorted, start=1):
        if str(item.issue_type or "").startswith("LLM判定："):
            continue
        selected.append((idx, item))
    results: Dict[int, Dict[str, str]] = {}
    if not selected:
        return results

    def _mk_item_payload(index_1_based: int, item: Finding) -> Dict[str, object]:
        keywords = _extract_checkpoint_keywords(item.snippet) if item.issue_type == "检查要点覆盖不足" else []
        context_cells: List[Tuple[str, str]] = []
        if item.sheet in wb.sheetnames and keywords:
            context_cells = _extract_context_cells(wb[item.sheet], keywords=keywords, limit=8)
        return {
            "id": index_1_based,
            "issue_type": item.issue_type,
            "rule_severity": item.severity,
            "sheet": item.sheet,
            "cell": item.cell or "",
            "excerpt": _safe_cell_text(item.snippet, 1200),
            "rule_basis": _safe_cell_text(item.basis, 1200),
            "rule_suggestion": _safe_cell_text(item.suggestion, 1200),
            "checkpoint_keywords": keywords,
            "context_cells": [{"cell": c, "text": t} for c, t in context_cells],
        }

    system_prompt = (
        "你是一名严格的IT审计/财务审计质量复核专家。\n"
        "你将收到一组“规则/启发式”识别的问题点（每条含：问题类型、严重级别、Sheet/单元格定位、原文摘录、判定依据、整改建议，以及可能的上下文单元格）。\n"
        "你的任务：逐条复核其是否成立、风险影响、需要补充的证据/程序、以及更合适的整改建议。\n"
        "要求：\n"
        "1) 不要泛泛而谈，要结合该条的摘录/上下文给出具体可执行建议（证据类型、样本总体/抽样基准、覆盖范围、职责分离、日志/台账/审批/协议等）。\n"
        "2) 如果你认为该条可能误报/信息不足，要明确说明需要补充哪些信息才能判断。\n"
        "3) 输出必须为严格JSON对象，格式为 {\"results\": [...]}。\n"
        "4) results 内每个元素对应输入的id，且必须包含字段：\n"
        "   - id: 整数\n"
        "   - status: \"pass\"/\"fail\"/\"unknown\"\n"
        "   - llm_validity: 成立/不成立/不确定（向后兼容）\n"
        "   - llm_severity: 高/中/低（向后兼容）\n"
        "   - severity: P0/P1/P2（结构化）\n"
        "   - conclusion: 一句话结论\n"
        "   - reasons: 字符串数组，2-5条要点\n"
        "   - evidence_refs: 数组，{sheet, cell_or_range, excerpt}。excerpt必须逐字来自摘录/上下文。fail时必填。\n"
        "   - llm_comment: 字符串（向后兼容）\n"
        "   - llm_missing_evidence: 字符串数组（向后兼容）\n"
        "   - llm_next_actions: 字符串数组（向后兼容）\n"
        "   - risk_type: 覆盖性/一致性/证据不足/方法性/逻辑性/跨字段一致性\n"
        "   - fix_suggestion: {missing_field, supplement_explanation, required_evidence_type}\n"
        "   - unknown_reason: 当status=unknown时必填，≥10字符\n"
        "5) 不要输出Markdown代码块，不要输出多余解释文字。\n"
    )

    for start in range(0, len(selected), max(1, int(batch_size))):
        chunk = selected[start : start + max(1, int(batch_size))]
        end = start + len(chunk)
        print(f"LLM复核进度: {start + 1}-{end}/{len(selected)}", flush=True)
        payload = [_mk_item_payload(idx, item) for idx, item in chunk]
        user_prompt = (
            "请对以下问题逐条进行复核，给出结构化结论。\n\n"
            f"输入问题（JSON）：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "请输出严格JSON对象，格式为 {\"results\": [...]}，其中 results 的每个元素必须包含字段：id, llm_validity, llm_severity, llm_comment, llm_missing_evidence, llm_next_actions。\n"
        )

        def _consume_results(objs: List[object]) -> None:
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                idx = obj.get("id")
                if not isinstance(idx, int):
                    continue
                # --- status 迁移 ---
                raw_status = str(obj.get("status", "")).strip()
                if raw_status == "成立":
                    status = "fail"
                elif raw_status == "不成立":
                    status = "pass"
                elif raw_status == "不确定":
                    status = "unknown"
                else:
                    status = raw_status or str(obj.get("llm_validity", "")).strip()
                    # 从 llm_validity 推断
                    if not status:
                        lv = str(obj.get("llm_validity", "")).strip()
                        if lv == "成立":
                            status = "fail"
                        elif lv == "不成立":
                            status = "pass"
                        elif lv == "不确定":
                            status = "unknown"
                # severity 迁移
                raw_sev = str(obj.get("severity") or obj.get("llm_severity", "")).strip()
                severity = _SEVERITY_FROM_CHINESE.get(raw_sev, raw_sev)
                # evidence_refs
                raw_refs = obj.get("evidence_refs")
                evidence_refs_list: List[dict] = []
                if isinstance(raw_refs, list) and raw_refs:
                    for ref in raw_refs:
                        if not isinstance(ref, dict):
                            continue
                        evidence_refs_list.append({
                            "sheet": str(ref.get("sheet", "")).strip(),
                            "cell_or_range": str(ref.get("cell_or_range", "")).strip(),
                            "attachment": str(ref.get("attachment", "")).strip(),
                            "excerpt": str(ref.get("excerpt", "")).strip(),
                        })
                # reasons
                reasons_raw = obj.get("reasons", [])
                reasons_list = [str(r).strip() for r in reasons_raw if r] if isinstance(reasons_raw, list) else []
                # fix_suggestion
                fix_sug = obj.get("fix_suggestion") or {}
                if not isinstance(fix_sug, dict):
                    fix_sug = {}
                results[idx] = {
                    "llm_validity": str(obj.get("llm_validity", "")).strip(),
                    "llm_severity": str(obj.get("llm_severity", "")).strip(),
                    "llm_comment": str(obj.get("llm_comment", "")).strip(),
                    "llm_missing_evidence": json.dumps(obj.get("llm_missing_evidence", []), ensure_ascii=False),
                    "llm_next_actions": json.dumps(obj.get("llm_next_actions", []), ensure_ascii=False),
                    # --- 新增结构化字段 ---
                    "llm_status": status,
                    "llm_severity_p": severity,
                    "llm_conclusion": str(obj.get("conclusion", "")).strip(),
                    "llm_reasons": json.dumps(reasons_list, ensure_ascii=False),
                    "llm_evidence_refs": json.dumps(evidence_refs_list, ensure_ascii=False),
                    "llm_risk_type": str(obj.get("risk_type", "")).strip(),
                    "llm_fix_suggestion": json.dumps(fix_sug, ensure_ascii=False),
                    "llm_unknown_reason": str(obj.get("unknown_reason", "")).strip(),
                }

        stage = "review_findings"
        parsed, last_error = _llm_request_json_list(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage=stage,
            max_attempts=3,
        )
        if parsed is not None:
            _consume_results(parsed)
            print(f"LLM复核完成: {start + 1}-{end}/{len(selected)}", flush=True)
        else:
            if len(chunk) > 1:
                _llm_stat(stage, "fallback_single", 1)
                for idx, item in chunk:
                    payload1 = [_mk_item_payload(idx, item)]
                    user_prompt1 = (
                        "请对以下问题逐条进行复核，给出结构化结论。\n\n"
                        f"输入问题（JSON）：\n{json.dumps(payload1, ensure_ascii=False, indent=2)}\n\n"
                        "请输出严格JSON对象，格式为 {\"results\": [...]}，其中 results 的每个元素必须包含字段：id, llm_validity, llm_severity, llm_comment, llm_missing_evidence, llm_next_actions。\n"
                    )
                    parsed1, err1 = _llm_request_json_list(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt1,
                        stage=stage,
                        max_attempts=3,
                    )
                    if parsed1 is not None:
                        _consume_results(parsed1)
                    else:
                        results[idx] = {
                            "llm_validity": "不确定",
                            "llm_severity": "",
                            "llm_comment": f"LLM调用失败: {err1}",
                            "llm_missing_evidence": "[]",
                            "llm_next_actions": "[]",
                            "llm_status": "unknown",
                            "llm_severity_p": "P2",
                            "llm_conclusion": "LLM调用失败",
                            "llm_reasons": "[]",
                            "llm_evidence_refs": "[]",
                            "llm_risk_type": "",
                            "llm_fix_suggestion": "{}",
                            "llm_unknown_reason": "LLM调用失败，无法复核",
                        }
            else:
                print(f"LLM复核失败: {start + 1}-{end}/{len(selected)}: {last_error}", flush=True)
                for idx, _ in chunk:
                    results[idx] = {
                        "llm_validity": "不确定",
                        "llm_severity": "",
                        "llm_comment": f"LLM调用失败: {last_error}",
                        "llm_missing_evidence": "[]",
                        "llm_next_actions": "[]",
                        "llm_status": "unknown",
                        "llm_severity_p": "P2",
                        "llm_conclusion": "LLM调用失败",
                        "llm_reasons": "[]",
                        "llm_evidence_refs": "[]",
                        "llm_risk_type": "",
                        "llm_fix_suggestion": "{}",
                        "llm_unknown_reason": "LLM调用失败，无法复核",
                    }

        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    return results


def _llm_judge_procedure_pair(
    client,
    model: str,
    standard_text: str,
    execution_text: str,
) -> Tuple[bool, Optional[bool], str, str]:
    def _clip(text: Optional[str], limit: int) -> str:
        s = (text or "").strip()
        if len(s) <= limit:
            return s
        return s[:limit] + "..."

    # 优先请求结构化 JSON 输出
    system_prompt_json = (
        "你是一名严格的审计质量复核专家。\n"
        "你的任务：对比A列的【标准审计程序描述】与C列的【实际执行程序】，判断执行是否满足标准的控制意图与关键要求。\n\n"
        "重要说明（用于降低误报）：\n"
        "1) 标准描述通常是“规范流程/参考口径”，不要求与执行描述在措辞、人名、部门称谓、文件标题完全一致。\n"
        "2) 访谈对象：若执行访谈对象属于同职能部门/同职责岗位，且能覆盖标准要验证的控制点，可视为符合；仅当职责明显不匹配或关键岗位未覆盖时才判不符合。\n"
        "3) 检查文件：若执行检查的制度/规范/流程文件与标准要求主题一致、适用范围一致或更高层级覆盖，可视为符合；仅当文件主题不相关或未覆盖关键控制要素时才判不符合。\n"
        "4) 结论表述：允许文字概括，但必须覆盖标准中的关键条件/核查点；若仅泛泛表述而未触及关键条件，则应判不符合或不确定。\n"
        "5) 无发生/不适用：若执行描述明确说明审计期间内该事项/活动未发生（如无迁移/无开发项目/无重大变更等），且给出总体为0或无发生的依据（例如项目清单/变更台账/发布记录/日志导出等），则可判pass并在理由中说明“不适用/总体为0”；若仅口头说明且缺少依据，则判unknown。\n\n"
        "判断规则：\n"
        "1. 标准描述中要求的审计动作，实际执行中是否包含\n"
        "2. 标准描述中指定的审计对象，实际执行中是否覆盖\n"
        "3. 标准描述中提出的具体条件，实际执行中是否满足\n"
        "4. 标准描述中要求获取的审计证据类型，实际执行中是否获取\n\n"
        "输出要求：必须输出严格JSON对象，格式为：\n"
        "{\"status\": \"pass\"|\"fail\"|\"unknown\", \"reason\": \"...\", "
        "\"evidence_refs\": [{\"cell\": \"...\", \"excerpt\": \"...\"}], "
        "\"unknown_reason\": \"...\"}\n"
        "evidence_refs: status=fail时必填，excerpt必须来自执行描述或标准描述的原文片段。"
        "status=unknown时unknown_reason必填（≥10字符）。"
    )
    # 旧格式 fallback prompt
    system_prompt_old = (
        "你是一名严格的审计质量复核专家。\n"
        "你的任务：对比A列的【标准审计程序描述】与C列的【实际执行程序】，判断执行是否满足标准的控制意图与关键要求。\n\n"
        "重要说明（用于降低误报）：\n"
        "1) 标准描述通常是“规范流程/参考口径”，不要求与执行描述在措辞、人名、部门称谓、文件标题完全一致。\n"
        "2) 访谈对象：若执行访谈对象属于同职能部门/同职责岗位，且能覆盖标准要验证的控制点，可视为符合；仅当职责明显不匹配或关键岗位未覆盖时才判不符合。\n"
        "3) 检查文件：若执行检查的制度/规范/流程文件与标准要求主题一致、适用范围一致或更高层级覆盖，可视为符合；仅当文件主题不相关或未覆盖关键控制要素时才判不符合。\n"
        "4) 结论表述：允许文字概括，但必须覆盖标准中的关键条件/核查点；若仅泛泛表述而未触及关键条件，则应判不符合或不确定。\n"
        "5) 无发生/不适用：若执行描述明确说明审计期间内该事项/活动未发生（如无迁移/无开发项目/无重大变更等），且给出总体为0或无发生的依据（例如项目清单/变更台账/发布记录/日志导出等），则可判【符合】并在理由中说明“不适用/总体为0”；若仅口头说明且缺少依据，则判【不确定】。\n\n"
        "判断规则：\n"
        "1. 标准描述中要求的审计动作，实际执行中是否包含\n"
        "2. 标准描述中指定的审计对象，实际执行中是否覆盖\n"
        "3. 标准描述中提出的具体条件，实际执行中是否满足\n"
        "4. 标准描述中要求获取的审计证据类型，实际执行中是否获取\n\n"
        "请只回答：【符合】或【不符合】或【不确定】\n"
        "然后换行写【理由：】后面跟简要说明。"
    )
    user_prompt = (
        "【标准审计程序描述 - A列】：\n"
        f"{_clip(standard_text, 900)}\n\n"
        "【实际执行程序 - C列】：\n"
        f"{_clip(execution_text, 900)}\n\n"
        "请判断C列执行程序是否符合A列标准要求。"
    )

    stage = "procedure_pair"
    # 优先尝试 JSON 输出
    try:
        answer = _llm_chat(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt_json},
                {"role": "user", "content": user_prompt},
            ],
            stage=stage,
            max_attempts=3,
            temperature=0.1,
            max_tokens=512,
        )
        answer = (answer or "").strip()
        parsed = _try_parse_json(answer)
        if isinstance(parsed, dict) and "status" in parsed:
            status_val = str(parsed.get("status", "")).strip()
            if status_val == "pass":
                is_match = True
            elif status_val == "fail":
                is_match = False
            else:
                is_match = None
            reason = str(parsed.get("reason", "")).strip() or str(parsed.get("unknown_reason", "")).strip()
            return True, is_match, reason, answer
    except Exception:
        pass

    # Fallback：使用旧 prompt 重试一次，再正则解析
    try:
        answer = _llm_chat(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt_old},
                {"role": "user", "content": user_prompt},
            ],
            stage=stage,
            max_attempts=2,
            temperature=0.1,
            max_tokens=512,
        )
        answer = (answer or "").strip()
        if "不符合" in answer:
            is_match = False
        elif "符合" in answer:
            is_match = True
        else:
            is_match = None

        if "理由：" in answer:
            reason = answer.split("理由：", 1)[1].strip()
        elif "理由:" in answer:
            reason = answer.split("理由:", 1)[1].strip()
        else:
            reason = answer
        reason = str(reason or "").strip("】 \t\n\r").strip()
        return True, is_match, reason, answer
    except Exception as e:
        return False, None, str(e), ""


def _llm_check_procedure_pairs(
    wb,
    target_sheets: Sequence[str],
    model: str,
    api_key: str,
    base_url: Optional[str],
    start_row: int = 5,
    sleep_seconds: float = 0.2,
) -> Tuple[Dict[str, object], List[Finding]]:
    from openpyxl.utils import get_column_letter

    client = _ensure_openai_client(api_key=api_key, base_url=base_url)

    skip_a_keywords = {
        "序号",
        "审计证据",
        "设计有效性测试结论",
        "执行有效性测试结论",
        "测试步骤",
        "抽样数量",
        "测试期间&样本总量",
        "样本记录（如果抽样数量>1，则在下表记录其余样本）",
        "缺陷评估",
        "是否发现异常",
        "缺陷描述",
        "标记注释",
    }
    skip_a_exact = {"n/a", "na", "不适用", "有效", "无"}
    skip_a_symbol = {"√", "×", "✓", "✗"}
    skip_c_exact = {"n/a", "na", "不适用"}

    def _execution_label(ws, header_row: Optional[int], execution_col: int) -> str:
        label = None
        if header_row:
            label = _get_cell_value(ws, f"{get_column_letter(execution_col)}{header_row + 1}")
            if not label:
                label = _get_cell_value(ws, f"{get_column_letter(execution_col)}{header_row}")
        return label or get_column_letter(execution_col)

    total = 0
    matched = 0
    failed = 0
    api_errors = 0
    skipped_a_empty = 0
    skipped_c_empty = 0
    skipped_ref = 0
    skipped_header = 0

    sheet_stats: Dict[str, Dict[str, int]] = {}
    result_details: List[Dict[str, Optional[str]]] = []
    issue_details: List[Dict[str, Optional[str]]] = []
    findings: List[Finding] = []

    keywords_like_procedure = ("询问", "访谈", "检查", "获取", "抽取", "观察", "复核", "比对", "重新执行", "分析", "确认", "审查")

    def _is_design_section(ws, header_row: Optional[int], standard_col: int) -> bool:
        if not header_row or standard_col <= 0:
            return False
        header_text = _get_cell_value(ws, f"{get_column_letter(standard_col)}{header_row}") or ""
        return "设计有效性" in header_text

    def _looks_password_design_standard(text: str) -> bool:
        t = (text or "").replace(" ", "").replace("\n", "").replace("\r", "")
        if not t:
            return False
        if "密码" not in t and "身份验证" not in t:
            return False
        if "设计" not in t and "设计有效性" not in t:
            return False
        return any(k in t for k in ("最短长度", "复杂", "到期", "锁定", "账户", "账号"))

    def _has_policy_evidence(text: str) -> bool:
        t = (text or "")
        return any(k in t for k in ("制度", "规程", "政策", "流程", "规定", "办法", "指引", "《", "<"))

    def _looks_no_occurrence_exec(execution_text: str) -> bool:
        t = (execution_text or "").replace(" ", "").replace("\n", "").replace("\r", "")
        if not t:
            return False
        time_markers = ("审计期间", "本期", "期间内", "测试期间", "本年度", "年度内")
        if not any(x in t for x in time_markers):
            return False
        if "未对此控制点进行测试" in t or "故未对" in t or "不适用" in t or t.lower().find("n/a") >= 0:
            return True
        action_markers = ("迁移", "开发", "重开发", "上线", "项目", "变更", "发布", "投产", "升级", "实施", "切换", "改造", "功能重开发")
        return re.search(r"(未|无)(发生|进行|开展|实施|发生过)?(.{0,18})(" + "|".join(action_markers) + ")", t) is not None

    def _standard_looks_conditional(standard_text: str) -> bool:
        t = (standard_text or "").replace(" ", "").replace("\n", "").replace("\r", "")
        if not t:
            return False
        markers = ("抽取", "样本", "期间", "迁移", "上线", "开发", "项目", "变更", "发布", "投产", "升级", "实施", "切换", "改造")
        return any(m in t for m in markers)

    def _classify_mismatch(standard_text: str, execution_text: str, reason: str) -> Tuple[str, str, str]:
        r = (reason or "").strip()
        r2 = r.replace(" ", "")
        if _looks_no_occurrence_exec(execution_text) and _standard_looks_conditional(standard_text):
            has_basis_evidence = any(k in (execution_text or "") for k in EVIDENCE_KEYWORDS) or any(
                k in (execution_text or "") for k in ("清单", "导出", "台账", "日志", "工单", "记录", "报告")
            )
            sev = "低" if has_basis_evidence else "中"
            return (
                "LLM判定：期间内无发生/不适用（不应按未执行判缺陷）",
                sev,
                "将该控制点按“不适用/总体为0”处理：在底稿中明确审计期间总体=0，并补充无发生依据（项目清单/变更台账/发布记录/工单/日志导出等）；如仅访谈说明，需补充可复核的清单或系统导出佐证。",
            )
        if any(k in r2 for k in ("访谈对象", "访谈人", "受访", "访谈人员")) and any(k in r2 for k in ("不一致", "不同", "不匹配")):
            return (
                "LLM判定：访谈对象/角色表述差异（可能等效）",
                "低",
                "在底稿中补充访谈对象岗位/职责与标准角色的对应关系（同部门/同职能可视为等效），并说明选择该人员的原因；如涉及关键岗位职责差异，补充关键角色访谈或邮件确认。",
            )
        if any(k in r2 for k in ("检查对象", "制度", "规范", "办法", "要求", "流程", "文件")) and any(k in r2 for k in ("不一致", "不同", "不匹配")):
            return (
                "LLM判定：制度/文件名称差异（需确认覆盖范围）",
                "低",
                "在底稿中说明所检查文件与标准要求的主题一致性（范围/适用系统/章节映射），必要时补充目录截图或关键条款摘录，证明已覆盖标准控制要点。",
            )
        if any(k in r2 for k in ("未满足", "未明确", "未确认", "未覆盖", "缺少", "不足", "没有")) and any(
            k in r2 for k in ("条件", "完善", "方案", "计划", "所有", "全部", "必须")
        ):
            return (
                "LLM判定：执行结论未覆盖标准关键条件",
                "中",
                "补充对标准关键条件的逐条确认记录（例如：抽取若干项目方案/计划验证、引用制度条款、说明样本范围/期间），并在结论中明确对应条件是否满足。",
            )
        return (
            "LLM判定：执行程序不符合标准审计程序",
            "高",
            "补充/修改执行程序以覆盖标准要求的审计动作、对象、条件与证据类型，并在底稿中保留可复核来源（截图/导出清单/日志台账/审批或协议等）。",
        )

    for sheet in target_sheets:
        if sheet not in wb.sheetnames:
            continue
        ws = wb[sheet]
        header_row, standard_col, execution_cols = _detect_layout(ws)
        if not standard_col or not execution_cols:
            continue

        execution_labels = {c: _execution_label(ws, header_row, c) for c in execution_cols}
        sheet_is_design = _is_design_section(ws, header_row, standard_col)
        sheet_stats[ws.title] = {
            "total": 0,
            "matched": 0,
            "failed": 0,
            "api_errors": 0,
            "skipped_a_empty": 0,
            "skipped_c_empty": 0,
            "skipped_ref": 0,
            "skipped_header": 0,
        }

        empty_streak = 0
        for row in range(max(1, int(start_row)), (ws.max_row or 0) + 1):
            a_text = _get_cell_value(ws, f"{get_column_letter(standard_col)}{row}")
            has_any_exec = any(_get_cell_value(ws, f"{get_column_letter(c)}{row}") is not None for c in execution_cols)

            if a_text is None and not has_any_exec:
                empty_streak += 1
                if empty_streak >= 30:
                    break
                continue
            empty_streak = 0

            if a_text is None:
                skipped_a_empty += 1
                sheet_stats[ws.title]["skipped_a_empty"] += 1
                continue

            a_compact = a_text.replace(" ", "").replace("\n", "").strip()
            if a_compact in skip_a_symbol or a_compact.lower() in skip_a_exact or a_compact in skip_a_keywords:
                skipped_header += 1
                sheet_stats[ws.title]["skipped_header"] += 1
                continue
            if (a_compact.startswith("*") and a_compact[1:].isdigit()) or (a_compact.startswith("#") and a_compact[1:].isdigit()):
                skipped_header += 1
                sheet_stats[ws.title]["skipped_header"] += 1
                continue
            if len(a_compact) <= 10 and all(ch.isalnum() or ch in "-_./" for ch in a_compact):
                skipped_header += 1
                sheet_stats[ws.title]["skipped_header"] += 1
                continue

            a_for_judge = a_text.strip()
            if not any(k in a_for_judge for k in keywords_like_procedure) and "•" not in a_for_judge and "。" not in a_for_judge:
                skipped_header += 1
                sheet_stats[ws.title]["skipped_header"] += 1
                continue

            for exec_col in execution_cols:
                c_text = _get_cell_value(ws, f"{get_column_letter(exec_col)}{row}")
                exec_label = execution_labels.get(exec_col) or get_column_letter(exec_col)
                if c_text is None:
                    skipped_c_empty += 1
                    sheet_stats[ws.title]["skipped_c_empty"] += 1
                    continue

                c_compact = c_text.replace(" ", "").replace("\n", "").strip()
                if c_compact.lower() in skip_c_exact:
                    skipped_ref += 1
                    sheet_stats[ws.title]["skipped_ref"] += 1
                    continue
                if len(c_compact) <= 12 and all(ch.isalnum() or ch in "-_./" for ch in c_compact):
                    skipped_ref += 1
                    sheet_stats[ws.title]["skipped_ref"] += 1
                    continue

                c_for_judge = c_text.strip()
                if len(a_for_judge) < 20 or len(c_for_judge) < 20:
                    skipped_header += 1
                    sheet_stats[ws.title]["skipped_header"] += 1
                    continue

                total += 1
                sheet_stats[ws.title]["total"] += 1

                standard_cell = f"{get_column_letter(standard_col)}{row}"
                execution_cell = f"{get_column_letter(exec_col)}{row}"

                success, is_match, reason, raw = _llm_judge_procedure_pair(
                    client=client,
                    model=model,
                    standard_text=a_for_judge,
                    execution_text=c_for_judge,
                )
                if success and (is_match is False or is_match is None):
                    if sheet_is_design and _looks_password_design_standard(a_for_judge) and _has_policy_evidence(c_for_judge):
                        is_match = True
                        reason = "设计有效性：已引用/获取制度/规程等文件作为证据；参数细节（最短长度/锁定等）与覆盖范围（账户类型）可作为完善建议，不作为实施有效性不足判定。"
                    if _looks_no_occurrence_exec(c_for_judge) and _standard_looks_conditional(a_for_judge):
                        is_match = True
                        reason = "执行说明审计期间内该事项未发生/总体为0，故该控制点不适用；建议在底稿中补充总体为0的可复核依据（清单/台账/日志导出等）。"
                result_label = "不确定"
                if success:
                    if is_match is True:
                        matched += 1
                        sheet_stats[ws.title]["matched"] += 1
                        result_label = "✓ 符合"
                    elif is_match is False:
                        failed += 1
                        sheet_stats[ws.title]["failed"] += 1
                        result_label = "✗ 不符合"
                    else:
                        failed += 1
                        sheet_stats[ws.title]["failed"] += 1
                        result_label = "不确定"
                else:
                    api_errors += 1
                    sheet_stats[ws.title]["api_errors"] += 1
                    result_label = "API错误"

                record = {
                    "sheet": ws.title,
                    "row": str(row),
                    "standard_cell": standard_cell,
                    "execution_cell": execution_cell,
                    "execution_label": exec_label,
                    "result": result_label,
                    "reason": _truncate(reason, 500) if reason else "",
                    "a_text": _truncate(a_for_judge, 800),
                    "c_text": _truncate(c_for_judge, 800),
                    "raw": _truncate(raw, 800) if raw else "",
                }
                result_details.append(record)
                if result_label in {"✗ 不符合", "API错误", "不确定"}:
                    issue_details.append(record)

                    ev_refs = [{
                        "sheet": ws.title,
                        "cell_or_range": execution_cell,
                        "excerpt": c_for_judge[:_EXCERPT_MAX_LEN],
                    }]
                    ev_refs = _verify_evidence_refs(ev_refs, ws)

                    if result_label == "✗ 不符合":
                        issue_type, sev, sug = _classify_mismatch(a_for_judge, c_for_judge, reason or raw or "")
                        # severity 迁移：高/中/低 → P0/P1/P2
                        sev_p = _SEVERITY_FROM_CHINESE.get(sev, sev)
                        # status: fail；如果 evidence_refs 验证后为空则降级
                        proc_status = "fail"
                        proc_unknown = ""
                        if not ev_refs:
                            proc_status = "unknown"
                            proc_unknown = "无法引用原始证据佐证该判定，降级为不确定"
                            sev_p = "P2"
                        # 构造分层 basis
                        basis_parts = []
                        if a_for_judge:
                            basis_parts.append("标准程序: " + _truncate(a_for_judge, 800))
                        if reason or raw:
                            basis_parts.append("LLM依据: " + _truncate(reason or raw, 2000))
                        if ev_refs:
                            basis_parts.append(
                                "引用: " + "; ".join(
                                    f"{r.get('cell_or_range', '')}: {r.get('excerpt', '')[:200]}"
                                    for r in ev_refs[:2] if r.get('excerpt')
                                )
                            )
                        if proc_unknown:
                            basis_parts.append("不确定原因: " + proc_unknown)
                        findings.append(
                            Finding(
                                issue_type=issue_type,
                                severity=sev_p,
                                sheet=ws.title,
                                cell=execution_cell,
                                snippet=_truncate(c_for_judge, 220),
                                basis=_truncate("\n".join(p for p in basis_parts if p), 3000),
                                suggestion=sug,
                                status=proc_status,
                                risk_type="一致性",
                                evidence_refs=json.dumps(ev_refs, ensure_ascii=False),
                                conclusion=f"执行程序与标准审计程序在 {execution_cell} 不一致",
                                reasons=json.dumps([_truncate(reason or raw, 300)] if (reason or raw) else [], ensure_ascii=False),
                                fix_suggestion_detail=json.dumps({"supplement_explanation": sug[:300]}, ensure_ascii=False),
                                unknown_reason=proc_unknown,
                            )
                        )
                    elif result_label == "API错误":
                        # LLM 调用超时/报错 → 无法自动判定，降级为 unknown
                        err_detail = reason or raw or "API请求超时或返回错误"
                        findings.append(
                            Finding(
                                issue_type="A-C对应性：LLM调用失败（需人工复核）",
                                severity="P1",
                                sheet=ws.title,
                                cell=execution_cell,
                                snippet=_truncate(c_for_judge, 220),
                                basis=_truncate(
                                    f"标准程序: {_truncate(a_for_judge, 400)}\n"
                                    f"LLM调用失败详情: {err_detail[:800]}\n"
                                    "无法自动判定「标准审计程序」与「实际执行程序」的对应性，"
                                    "请人工对比以下两项：\n"
                                    f"  A列（标准）: {_truncate(a_for_judge, 200)}\n"
                                    f"  C列（执行）: {_truncate(c_for_judge, 200)}",
                                    3000,
                                ),
                                suggestion=(
                                    "人工复核步骤：\n"
                                    "1) 对照A列标准审计程序，逐项检查C列执行是否覆盖要求的审计动作、对象、条件与证据类型；\n"
                                    '2) 如实际已覆盖，在底稿中补充「证据→核查点→结论」的对应说明；\n'
                                    "3) 如确实未覆盖，作为缺陷记录并提出整改。"
                                ),
                                status="unknown",
                                unknown_reason=f"LLM调用失败（{err_detail[:200]}），无法自动判定，需人工复核",
                                risk_type="证据不足",
                                evidence_refs=json.dumps(ev_refs, ensure_ascii=False),
                            )
                        )
                    else:
                        # result_label == "不确定"：LLM 返回了结果但无法明确判断
                        llm_partial = reason or raw or ""
                        findings.append(
                            Finding(
                                issue_type="A-C对应性：LLM无法判定（需人工确认）",
                                severity="P1",
                                sheet=ws.title,
                                cell=execution_cell,
                                snippet=_truncate(c_for_judge, 220),
                                basis=_truncate(
                                    f"标准程序: {_truncate(a_for_judge, 400)}\n"
                                    f"LLM分析: {_truncate(llm_partial, 800)}\n"
                                    f"LLM无法确认标准与执行是否一致，建议人工判断覆盖性。",
                                    3000,
                                ),
                                suggestion=(
                                    "人工确认步骤：\n"
                                    "1) 逐条核对：A列每个审计动作/对象/条件是否在C列中有对应描述；\n"
                                    "2) 关注差异点：访谈对象是否覆盖关键岗位、检查文件是否主题一致、"
                                    "是否有替代性程序覆盖同一控制点；\n"
                                    "3) 如确认不符合，补充完整执行程序；如确认符合，在底稿中写明对应关系。"
                                ),
                                status="unknown",
                                unknown_reason=f"LLM无法明确判定标准审计程序与实际执行程序的一致性"
                                + (f"（LLM分析: {_truncate(llm_partial, 150)}）" if llm_partial else ""),
                                risk_type="证据不足",
                                evidence_refs=json.dumps(ev_refs, ensure_ascii=False),
                            )
                        )

                if sleep_seconds and sleep_seconds > 0:
                    time.sleep(float(sleep_seconds))

    report: Dict[str, object] = {
        "model": model,
        "base_url": base_url or "",
        "total": total,
        "matched": matched,
        "failed": failed,
        "api_errors": api_errors,
        "skipped_a_empty": skipped_a_empty,
        "skipped_c_empty": skipped_c_empty,
        "skipped_ref": skipped_ref,
        "skipped_header": skipped_header,
        "sheet_stats": sheet_stats,
        "results": result_details,
        "issues": issue_details,
    }
    return report, findings


def _safe_cell_text(value: Optional[str], limit: int = 30000) -> str:
    if value is None:
        return ""
    s = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()
    s = s.strip("】").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "..."




def _merge_cell_duplicates(
    combined_rows,
    *,
    client=None,
    model="",
):
    """Merge rows that share the same (sheet, cell). LLM-first, deterministic fallback."""
    groups = {}
    for idx, row_data in enumerate(combined_rows):
        sheet = str(row_data.get("sheet", "")).strip()
        c = str(row_data.get("cell", "")).strip()
        if sheet and c and c != "-":
            groups.setdefault((sheet, c), []).append(idx)

    dup_keys = [(k, idxs) for k, idxs in groups.items() if len(idxs) >= 2]
    if not dup_keys:
        return combined_rows

    removed_indices = set()
    llm_ok = 0
    llm_fail = 0

    for (sheet, cell), idxs in dup_keys:
        items = [combined_rows[i] for i in idxs]
        merged = None
        if client is not None:
            merged = _try_llm_merge_cell(client, model, sheet, cell, items)
        if merged is not None:
            llm_ok += 1
        else:
            if client is not None:
                llm_fail += 1
            merged = _deterministic_merge_cell(items)

        # --- 合并新字段：取第一条的 status/risk_type/evidence_refs_summary/fix_suggestion/unknown_reason/needs_review ---
        # 优先保留 fail/P0 的；若全部 unknown 则取第一项
        best_status = "unknown"
        best_status_idx = 0
        sev_rank = {"P0": 0, "P1": 1, "P2": 2}
        best_sev_rank = 9
        for j, it in enumerate(items):
            st = str(it.get("status", ""))
            sv = str(it.get("severity", ""))
            r = sev_rank.get(sv, 9)
            if st == "fail" and r < best_sev_rank:
                best_sev_rank = r
                best_status = st
                best_status_idx = j
            elif st != "unknown" and best_status == "unknown":
                best_status = st
                best_status_idx = j
        merged_first = items[best_status_idx]
        combined_rows[idxs[0]] = {
            "sheet": sheet,
            "cell": cell,
            "excerpt": max((str(it.get("excerpt", "")) for it in items), key=len),
            "llm": items[0].get("llm", {}),
            "status": merged_first.get("status", "unknown"),
            "risk_type": merged_first.get("risk_type", ""),
            "evidence_refs_summary": merged_first.get("evidence_refs_summary", ""),
            "fix_suggestion": merged_first.get("fix_suggestion", ""),
            "unknown_reason": merged_first.get("unknown_reason", ""),
            "needs_review": merged_first.get("needs_review", ""),
            **merged,
        }
        for i in idxs[1:]:
            removed_indices.add(i)

    if not removed_indices:
        return combined_rows

    result = []
    for idx, row_data in enumerate(combined_rows):
        if idx in removed_indices:
            continue
        result.append(row_data)

    parts = [f"{len(dup_keys)} 组合并 => 减少 {len(removed_indices)} 条"]
    if llm_ok > 0:
        parts.append(f"LLM合并{llm_ok}组")
    if llm_fail > 0:
        parts.append(f"规则合并{llm_fail}组")
    print(f"同单元格合并: {'，'.join(parts)}", flush=True)
    return result


def _deterministic_merge_cell(items):
    sev_rank = {"P0": 0, "高": 0, "P1": 1, "中": 1, "P2": 2, "低": 2}
    best_sev = "P1"
    best_r = 999
    for it in items:
        r = sev_rank.get(str(it.get("severity", "P1")), 9)
        if r < best_r:
            best_r = r
            best_sev = str(it.get("severity", "P1"))

    seen = set()
    issue_parts = []
    for it in items:
        v = str(it.get("issue", "")).strip()
        if v and v not in seen:
            seen.add(v)
            issue_parts.append(v)

    sources = sorted({str(it.get("source", "")).strip() for it in items if str(it.get("source", "")).strip()})

    seen_b = set()
    basis_parts = []
    for it in items:
        v = str(it.get("basis", "")).strip()
        if v and v not in seen_b:
            seen_b.add(v)
            basis_parts.append(v)

    seen_s = set()
    sug_parts = []
    for it in items:
        v = str(it.get("suggestion", "")).strip()
        if v and v not in seen_s:
            seen_s.add(v)
            sug_parts.append(v)

    return {
        "source": " + ".join(sources) if sources else "多维度合并",
        "severity": best_sev,
        "issue": "；".join(issue_parts),
        "basis": "\n---\n".join(basis_parts),
        "suggestion": "\n".join(sug_parts),
    }


def _try_llm_merge_cell(client, model, sheet, cell, items):
    sp = (
        "你是一名IT审计底稿复核专家。请将同一单元格的多条问题合并为一条，去除冗余。\n\n"
        "合并规则：\n"
        "1) 本质相同的多条问题 → 合并issue和basis，保留最完整的描述\n"
        "2) severity取最严重的一条（高 > 中 > 低）\n"
        "3) 合并后source写为\"多维度合并\"\n\n"
        "你必须只输出一行JSON，格式如下（不要Markdown、不要解释）：\n"
        '{"results":[{"issue":"合并后问题描述","basis":"合并后判定依据","suggestion":"合并后整改建议","severity":"高"}]}'
    )
    items_json = [{"id": i, "source": str(it.get("source", "")), "severity": str(it.get("severity", "")),
                   "issue": str(it.get("issue", "")), "basis": str(it.get("basis", "")),
                   "suggestion": str(it.get("suggestion", ""))} for i, it in enumerate(items, start=1)]
    up = f"Sheet: {sheet}  单元格: {cell}\n\n请合并以下{len(items)}条问题：\n{json.dumps(items_json, ensure_ascii=False, indent=2)}"
    try:
        parsed, last_err = _llm_request_json_list(client=client, model=model, system_prompt=sp,
                                                    user_prompt=up, stage="merge_cell_duplicates", max_attempts=2)
        if parsed and isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            m = parsed[0]
            return {"source": "多维度合并", "severity": str(m.get("severity", items[0].get("severity", "中"))),
                    "issue": str(m.get("issue", items[0].get("issue", ""))),
                    "basis": str(m.get("basis", items[0].get("basis", ""))),
                    "suggestion": str(m.get("suggestion", items[0].get("suggestion", "")))}
        # Fallback: try to extract JSON from raw response via _llm_chat
        if last_err and "非JSON" in str(last_err):
            raw = _llm_chat(client=client, model=model, messages=[
                {"role": "system", "content": sp},
                {"role": "user", "content": up},
            ], stage="merge_cell_duplicates", max_attempts=1, temperature=0.05, max_tokens=512)
            data = _try_parse_json(raw)
            if isinstance(data, dict):
                data = data.get("results") or data.get("data") or data
            if isinstance(data, dict) and "issue" in data:
                return {"source": "多维度合并", "severity": str(data.get("severity", items[0].get("severity", "中"))),
                        "issue": str(data.get("issue", items[0].get("issue", ""))),
                        "basis": str(data.get("basis", items[0].get("basis", ""))),
                        "suggestion": str(data.get("suggestion", items[0].get("suggestion", "")))}
        if last_err:
            print(f"  LLM合并失败({sheet}/{cell}): {last_err[:120]}", flush=True)
    except Exception as e:
        print(f"  LLM合并异常({sheet}/{cell}): {str(e)[:120]}", flush=True)
    return None




def _write_report_txt(
    output_path: str,
    started_at: datetime,
    excel_path: str,
    checkpoints_path: Optional[str],
    attachments_preview_path: Optional[str],
    attachments_preview: Optional[Dict[str, object]],
    sheet_names: Sequence[str],
    target_sheets: Sequence[str],
    findings_sorted: Sequence[Finding],
    by_severity: Dict[str, int],
    by_type: Dict[str, int],
    actor_by_sheet: Dict[str, Dict[str, List[Tuple[str, str]]]],
    sheets_filter_applied: bool,
    llm_ac_report: Optional[Dict[str, object]] = None,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# IT一般控制测试底稿复核报告（规则/启发式）\n\n")
        f.write(f"**生成时间**: {started_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**文件路径**: {excel_path}\n")
        if checkpoints_path:
            f.write(f"**检查要点**: {checkpoints_path}\n")
        if attachments_preview_path:
            f.write(f"**附件预览清单**: {attachments_preview_path}\n")
            if attachments_preview:
                counts = attachments_preview.get("status_counts") or {}
                if isinstance(counts, dict) and counts:
                    pairs = [f"{k or '(空)'}={int(v or 0)}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))]
                    f.write(f"**附件状态统计**: {', '.join(pairs)}\n")
        f.write(f"**Sheet数量（全量）**: {len(sheet_names)}\n")
        f.write(f"**Sheet数量（本次检查）**: {len(target_sheets)}\n")
        if sheets_filter_applied:
            f.write(f"**本次检查Sheet**: {', '.join(target_sheets)}\n")
        f.write("\n")

        f.write("## 1. 复核依据（来源）\n")
        f.write("- 常见问题清单：特权账号、访问权限增删改、权限清查、密码策略、特权监控、高管权限、批处理、变更管理等典型缺陷\n")
        f.write("- 审计程序对应性思路：以“标准审计程序”对照“执行审计程序”，重点关注证据类型/样本框定/覆盖范围/职责分离\n\n")

        f.write("## 2. 汇总\n")
        f.write(f"- 总问题数: {len(findings_sorted)}\n")
        if by_severity:
            f.write("- 严重级别分布: " + ", ".join(f"{k}:{v}" for k, v in sorted(by_severity.items())) + "\n")
        if by_type:
            top_types = sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            f.write("- Top问题类型: " + ", ".join(f"{k}({v})" for k, v in top_types) + "\n")
        f.write("\n")

        if isinstance(LLM_CALL_STATS, dict) and LLM_CALL_STATS:
            f.write("## 2.1 LLM调用统计\n")
            f.write("|阶段|调用次数|成功|超时|限流|服务端|解析|上下文|其他|回退单条|\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for stage, st in sorted(LLM_CALL_STATS.items(), key=lambda kv: str(kv[0])):
                if not isinstance(st, dict):
                    continue
                f.write(
                    "|{stage}|{calls}|{ok}|{timeout}|{rate}|{server}|{parse}|{context}|{other}|{fallback}|\n".format(
                        stage=str(stage).replace("|", "｜"),
                        calls=int(st.get("calls", 0) or 0),
                        ok=int(st.get("ok", 0) or 0),
                        timeout=int(st.get("error_timeout", 0) or 0),
                        rate=int(st.get("error_rate_limit", 0) or 0),
                        server=int(st.get("error_server", 0) or 0),
                        parse=int(st.get("error_parse", 0) or 0),
                        context=int(st.get("error_context", 0) or 0),
                        other=int(st.get("error_other", 0) or 0),
                        fallback=int(st.get("fallback_single", 0) or 0),
                    )
                )
            f.write("\n")

        f.write("## 3. 问题清单（含定位）\n")
        f.write("|序号|严重级别|问题类型|Sheet|单元格|原文摘录|判定依据|整改建议|\n")
        f.write("|---:|---|---|---|---|---|---|---|\n")
        for idx, item in enumerate(findings_sorted, start=1):
            cell = item.cell or "-"
            snippet = _safe_cell_text(item.snippet).replace("|", "｜")
            basis = _safe_cell_text(item.basis).replace("|", "｜")
            suggestion = _safe_cell_text(item.suggestion).replace("|", "｜")
            f.write(f"|{idx}|{item.severity}|{item.issue_type}|{item.sheet}|{cell}|{snippet}|{basis}|{suggestion}|\n")

        f.write("\n")
        f.write("## 4. 关键角色候选（便于联动复核）\n")
        for sheet in ("SA-4c", "SA-5", "PM-5", "PM-6", "SA-12"):
            if sheet not in actor_by_sheet:
                continue
            admins = sorted({token for _, token in actor_by_sheet[sheet].get("admins", [])})
            executors = sorted({token for _, token in actor_by_sheet[sheet].get("executors", [])})
            if not admins and not executors:
                continue
            parts = []
            if admins:
                parts.append("管理员=" + ",".join(admins))
            if executors:
                parts.append("执行/审批/复核人=" + ",".join(executors))
            f.write(f"- {sheet}: " + "；".join(parts) + "\n")
        if llm_ac_report:
            f.write("\n")
            f.write("## 5. LLM对应性检查（A列标准 vs 执行列）\n")
            f.write(f"- 模型: {llm_ac_report.get('model', '')}\n")
            f.write(f"- 接口: {llm_ac_report.get('base_url', '')}\n")
            f.write(f"- 总计检查: {llm_ac_report.get('total', 0)}\n")
            f.write(f"- 符合: {llm_ac_report.get('matched', 0)}\n")
            f.write(f"- 不符合: {llm_ac_report.get('failed', 0)}\n")
            f.write(f"- API错误: {llm_ac_report.get('api_errors', 0)}\n")
            f.write(f"- 跳过（执行为空）: {llm_ac_report.get('skipped_c_empty', 0)}\n")
            f.write(f"- 跳过（标准为空）: {llm_ac_report.get('skipped_a_empty', 0)}\n")
            f.write(f"- 跳过（引用/编号）: {llm_ac_report.get('skipped_ref', 0)}\n")
            f.write(f"- 跳过（标题/分类）: {llm_ac_report.get('skipped_header', 0)}\n")
            issues = llm_ac_report.get("issues") or []
            if isinstance(issues, list) and issues:
                f.write("\n")
                f.write("### 5.1 问题明细（不符合/API错误/不确定）\n")
                f.write("|Sheet|Row|标准单元格|执行单元格|执行对象|结果|理由|\n")
                f.write("|---|---:|---|---|---|---|---|\n")
                for item in issues:
                    if not isinstance(item, dict):
                        continue
                    f.write(
                        "|{sheet}|{row}|{standard_cell}|{execution_cell}|{execution_label}|{result}|{reason}|\n".format(
                            sheet=str(item.get("sheet", "")).replace("|", "｜"),
                            row=str(item.get("row", "")).replace("|", "｜"),
                            standard_cell=str(item.get("standard_cell", "")).replace("|", "｜"),
                            execution_cell=str(item.get("execution_cell", "")).replace("|", "｜"),
                            execution_label=str(item.get("execution_label", "")).replace("|", "｜"),
                            result=str(item.get("result", "")).replace("|", "｜"),
                            reason=_safe_cell_text(item.get("reason", "")).replace("|", "｜"),
                        )
                    )


def _write_report_xlsx(
    output_path: str,
    started_at: datetime,
    excel_path: str,
    checkpoints_path: Optional[str],
    attachments_preview_path: Optional[str],
    attachments_preview: Optional[Dict[str, object]],
    sheet_names: Sequence[str],
    target_sheets: Sequence[str],
    findings_sorted: Sequence[Finding],
    by_severity: Dict[str, int],
    by_type: Dict[str, int],
    actor_by_sheet: Dict[str, Dict[str, List[Tuple[str, str]]]],
    sheets_filter_applied: bool,
    llm_results: Optional[Dict[int, Dict[str, str]]] = None,
    llm_ac_report: Optional[Dict[str, object]] = None,
    *,
    client=None,
    model: str,
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    src_wb = None
    try:
        src_wb = openpyxl.load_workbook(excel_path, data_only=False)
    except Exception:
        src_wb = None

    def _safe_cell_text_multiline(value: Optional[str], limit: int = 30000) -> str:
        if value is None:
            return ""
        s = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(s) <= limit:
            return s
        return s[:limit] + "..."

    def _build_evidence_refs_summary_from_ac(it: dict) -> str:
        """Build an evidence_refs_summary string from an A-C correspondence issue dict.

        A-C issues come from _llm_check_procedure_pairs and have execution_cell, c_text,
        and optionally reason. We construct a summary similar to what Finding objects produce.
        """
        parts: List[str] = []
        cell = str(it.get("execution_cell", "") or "").strip()
        c_text = str(it.get("c_text", "") or "").strip()
        sheet = str(it.get("sheet", "") or "").strip()
        reason = str(it.get("reason", "") or "").strip()

        if cell:
            head = cell
            if sheet:
                head = f"{sheet}!{cell}" if cell else sheet
            line = f"[{head}]"
            if c_text:
                line += f"\n    摘录: {c_text[:_EXCERPT_MAX_LEN]}"
            parts.append(line)
        if not parts and reason:
            parts.append(f"[依据] {reason[:_EXCERPT_MAX_LEN]}")
        return "\n".join(parts)

    def _extract_cell_refs(text: str) -> List[str]:
        if not text:
            return []
        parts = [p.strip().upper() for p in re.split(r"[,\s]+", str(text)) if p.strip()]
        out: List[str] = []
        seen = set()
        for p in parts:
            if not re.match(r"^[A-Z]{1,3}\d{1,7}$", p):
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _full_excerpt(sheet_name: str, cell_field: Optional[str], fallback: Optional[str]) -> str:
        if not src_wb or not sheet_name or sheet_name not in getattr(src_wb, "sheetnames", []):
            return _safe_cell_text_multiline(fallback)
        ws_src = src_wb[sheet_name]
        refs = _extract_cell_refs(cell_field or "")
        parts: List[str] = []
        if refs:
            for ref in refs[:10]:
                t = _get_cell_value(ws_src, ref)
                if not t:
                    continue
                parts.append(f"{ref}: {t}")
        if parts:
            return _safe_cell_text_multiline("\n".join(parts))
        if cell_field and str(cell_field).strip() and str(cell_field).strip() not in {"-", "—"}:
            t2 = _get_cell_value(ws_src, str(cell_field).strip())
            if t2:
                return _safe_cell_text_multiline(t2)
        return _safe_cell_text_multiline(fallback)

    wb = Workbook()
    wb.remove(wb.active)

    wrap = Alignment(wrap_text=True, vertical="top")
    header_font = Font(bold=True)

    ws_sum = wb.create_sheet("汇总")
    ws_sum.column_dimensions["A"].width = 20
    ws_sum.column_dimensions["B"].width = 52
    ws_sum.column_dimensions["C"].width = 20
    ws_sum.column_dimensions["D"].width = 42
    ws_sum.column_dimensions["E"].width = 10

    # ── 标题 ──
    ws_sum.merge_cells("A1:D1")
    title_cell = ws_sum["A1"]
    title_cell.value = "IT一般控制测试底稿复核报告"
    title_cell.font = Font(bold=True, size=14)

    # ── 基本信息 ──
    r = 3
    ws_sum.merge_cells(f"A{r}:D{r}")
    ws_sum[f"A{r}"] = "基本信息"
    ws_sum[f"A{r}"].font = Font(bold=True, size=11)
    r += 1

    meta = [
        ("生成时间", started_at.strftime("%Y-%m-%d %H:%M:%S")),
        ("文件路径", excel_path),
        ("检查要点", checkpoints_path or "（未指定）"),
        ("Sheet数量（本次/全量）", f"{len(target_sheets)} / {len(sheet_names)}"),
        ("本次检查Sheet", ", ".join(target_sheets) if sheets_filter_applied else "全部"),
        ("附件预览清单", attachments_preview_path or "（未指定）"),
        ("总问题数", str(len(findings_sorted))),
    ]
    for label, value in meta:
        ws_sum[f"A{r}"] = label
        ws_sum[f"B{r}"] = value
        ws_sum[f"A{r}"].font = Font(bold=False)
        ws_sum[f"B{r}"].alignment = wrap
        r += 1

    # ── 问题统计 ──
    r += 1
    ws_sum.merge_cells(f"A{r}:D{r}")
    ws_sum[f"A{r}"] = "问题统计"
    ws_sum[f"A{r}"].font = Font(bold=True, size=11)
    r += 1

    # 严重级别
    ws_sum[f"A{r}"] = "严重级别"
    ws_sum[f"B{r}"] = "数量"
    ws_sum[f"A{r}"].font = header_font
    ws_sum[f"B{r}"].font = header_font
    r += 1
    _summary_sev_start_row = r
    for sev, cnt in sorted(by_severity.items()):
        ws_sum[f"A{r}"] = sev
        ws_sum[f"B{r}"] = cnt
        r += 1

    r += 1
    # Top问题类型
    type_start = r
    ws_sum.merge_cells(f"A{type_start}:D{type_start}")
    ws_sum[f"A{type_start}"] = "Top问题类型"
    ws_sum[f"A{type_start}"].font = Font(bold=True, size=11)
    r += 1
    ws_sum[f"A{r}"] = "问题类型"
    ws_sum[f"B{r}"] = "数量"
    ws_sum[f"A{r}"].font = header_font
    ws_sum[f"B{r}"].font = header_font
    r += 1
    _summary_type_start_row = r
    for issue_type, cnt in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        ws_sum[f"A{r}"] = issue_type
        ws_sum[f"B{r}"] = cnt
        r += 1

    ws_issues = wb.create_sheet("问题清单")
    headers = [
        "Sheet",
        "来源",
        "状态",
        "严重级别",
        "风险类型",
        "问题类型/结果",
        "单元格",
        "原文全文",
        "证据引用",
        "判定依据/理由",
        "整改建议(结构化)",
        "不确定原因",
        "需复核",
    ]
    include_llm_review_cols = bool(os.getenv("INCLUDE_LLM_REVIEW_COLS", "").strip())
    if llm_results is not None and include_llm_review_cols:
        headers.extend(["LLM成立性", "LLM复核严重级别", "LLM复核意见", "LLM缺失证据", "LLM下一步动作"])
    for c, h in enumerate(headers, start=1):
        cell = ws_issues.cell(row=1, column=c, value=h)
        cell.font = header_font
        cell.alignment = Alignment(vertical="top")
    ws_issues.freeze_panes = "A2"

    widths = [14, 12, 8, 10, 14, 28, 10, 60, 100, 100, 60, 30, 10]
    if llm_results is not None and include_llm_review_cols:
        widths.extend([12, 14, 70, 50, 50])
    for idx, w in enumerate(widths, start=1):
        ws_issues.column_dimensions[get_column_letter(idx)].width = w

    def _sev_rank(sev: str) -> int:
        # 同时支持新 P0/P1/P2 和旧 高/中/低
        s = str(sev or "").strip()
        return {"P0": 0, "高": 0, "P1": 1, "中": 1, "P2": 2, "低": 2}.get(s, 9)

    combined_rows: List[Dict[str, object]] = []
    for idx, item in enumerate(findings_sorted, start=1):
        llm = llm_results.get(idx) if llm_results is not None else None
        # 解析 evidence_refs JSON 字符串
        try:
            refs_list = json.loads(item.evidence_refs) if item.evidence_refs else []
        except Exception:
            refs_list = []
        # 多行格式：每个 ref 一行，包含 sheet、cell、attachment、excerpt
        # （不截断 excerpt，让审计员能完整看到原文）
        refs_lines: List[str] = []
        for ref in refs_list[:6]:
            if not isinstance(ref, dict):
                continue
            parts: List[str] = []
            sheet_label = str(ref.get("sheet", "") or "").strip()
            cell_label = str(ref.get("cell_or_range", "") or "").strip()
            if sheet_label and sheet_label != item.sheet:
                parts.append(sheet_label)
            if cell_label:
                parts.append(cell_label)
            attach = str(ref.get("attachment", "") or "").strip()
            excerpt = str(ref.get("excerpt", "") or "").strip()
            head = "/".join(parts) if parts else "?"
            line = f"[{head}]"
            if attach:
                line += f" 附件={attach}"
            if excerpt:
                line += f"\n    摘录: {excerpt}"
            refs_lines.append(line)
        refs_summary = "\n".join(refs_lines)
        # 解析 fix_suggestion_detail
        try:
            fix_sug = json.loads(item.fix_suggestion_detail) if item.fix_suggestion_detail else {}
        except Exception:
            fix_sug = {}
        fix_sug_parts = []
        if isinstance(fix_sug, dict):
            if fix_sug.get("missing_field"):
                fix_sug_parts.append(f"缺: {fix_sug['missing_field'][:200]}")
            if fix_sug.get("supplement_explanation"):
                fix_sug_parts.append(f"补: {fix_sug['supplement_explanation'][:200]}")
            if fix_sug.get("required_evidence_type"):
                fix_sug_parts.append(f"需证据: {fix_sug['required_evidence_type'][:200]}")
        fix_sug_text = " | ".join(fix_sug_parts) if fix_sug_parts else _safe_cell_text(item.suggestion)
        combined_rows.append(
            {
                "sheet": item.sheet,
                "source": "规则/启发式",
                "status": item.status,
                "severity": item.severity,
                "risk_type": item.risk_type,
                "issue": item.issue_type,
                "cell": item.cell or "-",
                "excerpt": _full_excerpt(item.sheet, item.cell, item.snippet),
                "evidence_refs_summary": refs_summary,
                "basis": _safe_cell_text(item.basis),
                "fix_suggestion": fix_sug_text,
                "unknown_reason": item.unknown_reason,
                "needs_review": "是" if item.needs_review else "",
                "llm": llm or {},
            }
        )

    if llm_ac_report:
        issues = llm_ac_report.get("issues") or []
        if isinstance(issues, list):
            for it in issues:
                if not isinstance(it, dict):
                    continue
                result = str(it.get("result", "") or "").strip()
                # Use internal P0/P1/P2 severity (consistent with Finding dataclass)
                sev = "P1"
                if "✗" in result or "不符合" in result:
                    sev = "P0"
                elif "API" in result:
                    sev = "P1"
                elif "不确定" in result:
                    sev = "P1"
                combined_rows.append(
                    {
                        "sheet": str(it.get("sheet", "") or "").strip(),
                        "source": "LLM对应性",
                        "status": "fail" if "不符合" in result else ("unknown" if "不确定" in result or "API" in result else "pass"),
                        "severity": sev,
                        "risk_type": "一致性",
                        "issue": f"A-C对应性：{result}" if result else "A-C对应性：问题",
                        "cell": str(it.get("execution_cell", "") or "").strip() or "-",
                        "excerpt": _full_excerpt(
                            str(it.get("sheet", "") or "").strip(),
                            str(it.get("execution_cell", "") or "").strip(),
                            it.get("c_text", "") or "",
                        ),
                        "evidence_refs_summary": _build_evidence_refs_summary_from_ac(it),
                        "basis": _safe_cell_text(it.get("reason", "") or ""),
                        "fix_suggestion": "对照标准审计程序，补充/修订执行步骤与证据，写清「证据→核查点→结论」的对应关系。",
                        "unknown_reason": "",
                        "needs_review": "",
                        "llm": {},
                    }
                )

    # src_wb no longer needed after building combined_rows
    if src_wb is not None:
        try:
            src_wb.close()
        except Exception:
            pass
        src_wb = None

    # Merge duplicate issues pointing to the same cell within the same sheet
    combined_rows = _merge_cell_duplicates(combined_rows, client=client, model=model)

    # ── 交叉验证（补充）：合并后再次核查 ──
    # 主要的 _cross_validate_finding 已在 generate_report 中对 Finding 对象运行，
    # 此处对 combined_rows 中的 P0+fail/unknown+无证据 做补充检查。
    for row_data in combined_rows:
        status = str(row_data.get("status", ""))
        sev = str(row_data.get("severity", ""))
        # 统一检查 P0（含中文"高"）的 fail/unknown + 无证据
        if sev in ("P0", "高") and status in ("fail", "unknown"):
            ev_summary = str(row_data.get("evidence_refs_summary", "") or "").strip()
            if not ev_summary:
                row_data["needs_review"] = "是"

    def _cell_sort_key(cell):
        if not cell or cell == "-":
            return ("Z", 10**9)
        first = cell.split(",")[0].strip().upper()
        m = re.match(r"^([A-Z]+)(\d+)$", first)
        if not m:
            return ("Z", 10**9)
        return (m.group(1), int(m.group(2)))

    combined_rows.sort(
        key=lambda rd: (
            str(rd.get("sheet", "")),
            _cell_sort_key(str(rd.get("cell", ""))),
            _sev_rank(str(rd.get("severity", ""))),
        )
    )

    # Recompute stats from merged combined_rows and update summary cells
    merged_by_sev: Dict[str, int] = defaultdict(int)
    merged_by_typ: Dict[str, int] = defaultdict(int)
    for row_data in combined_rows:
        merged_by_sev[_SEVERITY_DISPLAY.get(str(row_data.get("severity", "")), str(row_data.get("severity", "")))] += 1
        merged_by_typ[str(row_data.get("issue", ""))] += 1
    ws_sum["B10"] = str(len(combined_rows))
    r_upd = _summary_sev_start_row
    for sev, cnt in sorted(merged_by_sev.items()):
        ws_sum[f"A{r_upd}"] = sev
        ws_sum[f"B{r_upd}"] = cnt
        r_upd += 1
    # Clear stale severity rows
    while ws_sum[f"A{r_upd}"].value and str(ws_sum[f"A{r_upd}"].value).strip() in dict(by_severity):
        ws_sum[f"A{r_upd}"] = ""
        ws_sum[f"B{r_upd}"] = ""
        r_upd += 1
    r_upd = _summary_type_start_row
    for issue_type, cnt in sorted(merged_by_typ.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        ws_sum[f"A{r_upd}"] = issue_type
        ws_sum[f"B{r_upd}"] = cnt
        r_upd += 1

    rr = 2
    for row_data in combined_rows:
        sev_disp = _SEVERITY_DISPLAY.get(str(row_data.get("severity", "")).strip(), str(row_data.get("severity", "")))
        row = [
            row_data.get("sheet", ""),
            row_data.get("source", ""),
            row_data.get("status", ""),
            sev_disp,
            row_data.get("risk_type", ""),
            row_data.get("issue", ""),
            row_data.get("cell", ""),
            row_data.get("excerpt", ""),
            row_data.get("evidence_refs_summary", ""),
            row_data.get("basis", ""),
            row_data.get("fix_suggestion", ""),
            row_data.get("unknown_reason", ""),
            row_data.get("needs_review", ""),
        ]
        if llm_results is not None and include_llm_review_cols:
            llm = row_data.get("llm") if isinstance(row_data.get("llm"), dict) else {}
            row.extend([
                str(llm.get("llm_validity", "")),
                str(llm.get("llm_severity", "")),
                str(llm.get("llm_comment", "")),
                str(llm.get("llm_missing_evidence", "")),
                str(llm.get("llm_next_actions", "")),
            ])
        for cc, value in enumerate(row, start=1):
            cell = ws_issues.cell(row=rr, column=cc, value=value)
            if cc >= 6:
                cell.alignment = wrap
            else:
                cell.alignment = Alignment(vertical="top")
        rr += 1

    # ── 关键角色候选 ──
    r += 2
    ws_sum.merge_cells(f"A{r}:C{r}")
    ws_sum[f"A{r}"] = "关键角色候选（便于联动复核）"
    ws_sum[f"A{r}"].font = Font(bold=True, size=11)
    r += 1
    ws_sum[f"A{r}"] = "Sheet"
    ws_sum[f"B{r}"] = "管理员候选"
    ws_sum[f"C{r}"] = "执行/审批/复核人候选"
    for col in ("A", "B", "C"):
        ws_sum[f"{col}{r}"].font = header_font
        ws_sum[f"{col}{r}"].alignment = Alignment(vertical="top")
    ws_sum.column_dimensions["C"].width = 70
    r += 1

    roles_r = r
    for sheet in ("SA-4c", "SA-5", "PM-5", "PM-6", "SA-12"):
        if sheet not in actor_by_sheet:
            continue
        admins = sorted({token for _, token in actor_by_sheet[sheet].get("admins", [])})
        executors = sorted({token for _, token in actor_by_sheet[sheet].get("executors", [])})
        if not admins and not executors:
            continue
        ws_sum[f"A{roles_r}"] = sheet
        ws_sum[f"B{roles_r}"] = ", ".join(admins)
        ws_sum[f"C{roles_r}"] = ", ".join(executors)
        ws_sum[f"B{roles_r}"].alignment = wrap
        ws_sum[f"C{roles_r}"].alignment = wrap
        roles_r += 1
    r = max(roles_r, r)

    # ── LLM调用统计 ──
    r += 1
    llm_headers = ["阶段", "调用次数", "成功", "超时", "限流", "服务端", "解析", "上下文", "其他", "回退单条"]
    ws_sum.merge_cells(f"A{r}:J{r}")
    ws_sum[f"A{r}"] = "LLM调用统计"
    ws_sum[f"A{r}"].font = Font(bold=True, size=11)
    r += 1
    for c, h in enumerate(llm_headers, start=1):
        cell = ws_sum.cell(row=r, column=c, value=h)
        cell.font = header_font
        cell.alignment = Alignment(vertical="top")
    r += 1

    if isinstance(LLM_CALL_STATS, dict) and LLM_CALL_STATS:
        for stage, st in sorted(LLM_CALL_STATS.items(), key=lambda kv: str(kv[0])):
            if not isinstance(st, dict):
                continue
            row_llm = [
                str(stage),
                int(st.get("calls", 0) or 0),
                int(st.get("ok", 0) or 0),
                int(st.get("error_timeout", 0) or 0),
                int(st.get("error_rate_limit", 0) or 0),
                int(st.get("error_server", 0) or 0),
                int(st.get("error_parse", 0) or 0),
                int(st.get("error_context", 0) or 0),
                int(st.get("error_other", 0) or 0),
                int(st.get("fallback_single", 0) or 0),
            ]
            for cc, v in enumerate(row_llm, start=1):
                cell = ws_sum.cell(row=r, column=cc, value=v)
                cell.alignment = Alignment(vertical="top")
            r += 1

    # ── LLM对应性统计 ──
    if llm_ac_report:
        r += 1
        ws_sum.merge_cells(f"A{r}:D{r}")
        ws_sum[f"A{r}"] = "LLM对应性统计（标准审计程序 vs 执行审计程序）"
        ws_sum[f"A{r}"].font = Font(bold=True, size=11)
        r += 1

        ws_sum[f"A{r}"] = "模型"
        ws_sum[f"B{r}"] = str(llm_ac_report.get("model", ""))
        r += 1
        ws_sum[f"A{r}"] = "接口"
        ws_sum[f"B{r}"] = str(llm_ac_report.get("base_url", ""))
        r += 1

        pairs = [
            ("总计检查", "total"),
            ("符合", "matched"),
            ("不符合/不确定", "failed"),
            ("API错误", "api_errors"),
            ("跳过（执行为空）", "skipped_c_empty"),
            ("跳过（标准为空）", "skipped_a_empty"),
            ("跳过（引用/编号）", "skipped_ref"),
            ("跳过（标题/分类）", "skipped_header"),
        ]
        for label, key in pairs:
            ws_sum[f"A{r}"] = label
            ws_sum[f"B{r}"] = int(llm_ac_report.get(key, 0) or 0)
            r += 1

        r += 1
        headers = ["Sheet", "检查", "符合", "不符合/不确定", "API错误", "执行空", "标准空", "引用", "标题"]
        for c, h in enumerate(headers, start=1):
            cell = ws_sum.cell(row=r, column=c, value=h)
            cell.font = header_font
            cell.alignment = Alignment(vertical="top")
        r += 1

        sheet_stats = llm_ac_report.get("sheet_stats") or {}
        if isinstance(sheet_stats, dict):
            for name, st in sheet_stats.items():
                if not isinstance(st, dict):
                    continue
                ws_sum.cell(row=r, column=1, value=str(name)).alignment = Alignment(vertical="top")
                ws_sum.cell(row=r, column=2, value=int(st.get("total", 0) or 0)).alignment = Alignment(vertical="top")
                ws_sum.cell(row=r, column=3, value=int(st.get("matched", 0) or 0)).alignment = Alignment(vertical="top")
                ws_sum.cell(row=r, column=4, value=int(st.get("failed", 0) or 0)).alignment = Alignment(vertical="top")
                ws_sum.cell(row=r, column=5, value=int(st.get("api_errors", 0) or 0)).alignment = Alignment(vertical="top")
                ws_sum.cell(row=r, column=6, value=int(st.get("skipped_c_empty", 0) or 0)).alignment = Alignment(vertical="top")
                ws_sum.cell(row=r, column=7, value=int(st.get("skipped_a_empty", 0) or 0)).alignment = Alignment(vertical="top")
                ws_sum.cell(row=r, column=8, value=int(st.get("skipped_ref", 0) or 0)).alignment = Alignment(vertical="top")
                ws_sum.cell(row=r, column=9, value=int(st.get("skipped_header", 0) or 0)).alignment = Alignment(vertical="top")
                r += 1

    wb.save(output_path)


def generate_report(
    excel_path: str,
    output_path: str,
    checkpoints_path: Optional[str] = None,
    attachments_preview_path: Optional[str] = None,
    sheets: Optional[Sequence[str]] = None,
) -> None:
    llm_ac_start_row = 5
    llm_ac_sleep_seconds = 0.2
    llm_batch_size = 6
    llm_sleep_seconds = 0.2
    started_at = datetime.now()
    if not os.path.exists(excel_path):
        raise FileNotFoundError(excel_path)

    checkpoints_by_sheet = load_checkpoints_xlsx(checkpoints_path) if checkpoints_path else {}
    checkpoints_by_norm: Dict[str, List[str]] = defaultdict(list)
    if checkpoints_by_sheet:
        for k, items in checkpoints_by_sheet.items():
            nk = _normalize_sheet_id(k)
            for it in items:
                checkpoints_by_norm[nk].append(it)

    wb = openpyxl.load_workbook(excel_path, data_only=False)
    sheet_names = list(wb.sheetnames)
    target_sheets = list(sheet_names)
    if sheets:
        wanted = [str(s).strip() for s in sheets if str(s).strip()]
        missing = [s for s in wanted if s not in sheet_names]
        if missing:
            raise ValueError(f"指定的Sheet不存在: {missing}. 可用Sheet: {sheet_names}")
        target_sheets = wanted

    attachments_preview: Optional[Dict[str, object]] = None
    if attachments_preview_path:
        attachments_preview = load_attachments_preview_xlsx(attachments_preview_path)

    resolved_key, resolved_base_url, resolved_model = resolve_llm_config()
    if not resolved_key:
        raise RuntimeError(
            "未配置LLM的API Key（请在脚本同目录 .env 中设置 CHECKER_API_KEY/LLM_API_KEY/API_KEY/OPENAI_API_KEY，或设置对应环境变量）"
        )
    client = _ensure_openai_client(api_key=str(resolved_key), base_url=resolved_base_url)

    findings: List[Finding] = []
    actor_by_sheet: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}

    def _process_single_sheet(name: str) -> Tuple[str, List[Finding], Dict[str, List[Tuple[str, str]]]]:
        """Process one sheet: rule checks + LLM checks. Runs in a thread for parallelism.

        Each thread loads its own openpyxl workbook to avoid thread-safety issues
        with shared internal caches (string table, style dicts, etc.).
        """
        thread_wb = openpyxl.load_workbook(excel_path, data_only=False)
        try:
            ws = thread_wb[name]
            sheet_findings: List[Finding] = []
            if attachments_preview:
                sheet_findings.extend(_check_attachment_references(name, ws, attachments_preview))
                sheet_findings.extend(
                    _llm_check_evidence_vs_steps(
                        client=client,
                        model=str(resolved_model),
                        ws_title=name,
                        ws=ws,
                        attachments_preview=attachments_preview,
                        batch_size=6,
                        sleep_seconds=0.2,
                    )
                )
            if checkpoints_by_sheet:
                cps = checkpoints_by_sheet.get(name) or checkpoints_by_norm.get(_normalize_sheet_id(name)) or []
                if cps:
                    sheet_findings.extend(
                        _llm_check_sheet_by_checkpoints(
                            client=client,
                            model=str(resolved_model),
                            ws_title=name,
                            ws=ws,
                            checkpoints=cps,
                            attachments_preview=attachments_preview,
                            batch_size=6,
                            sleep_seconds=0.2,
                        )
                    )
            sheet_findings.extend(_check_sheet_scope(name, ws))
            sheet_findings.extend(_check_procedure_pairs(name, ws))
            actors = _extract_actor_candidates(ws)
            return name, sheet_findings, actors
        finally:
            thread_wb.close()

    with ThreadPoolExecutor(max_workers=min(4, len(target_sheets))) as executor:
        futures = {executor.submit(_process_single_sheet, name): name for name in target_sheets}
        for future in as_completed(futures):
            name, sheet_findings, actors = future.result()
            findings.extend(sheet_findings)
            actor_by_sheet[name] = actors

    sa_admins = {token for _, token in actor_by_sheet.get("SA-4c", {}).get("admins", [])}
    sa5_executors = {token for _, token in actor_by_sheet.get("SA-5", {}).get("executors", [])}
    if sa_admins and sa5_executors and sa_admins.isdisjoint(sa5_executors):
        findings.append(
            Finding(
                issue_type="跨测试点特权用户可能遗漏/结论联动不足",
                severity="P1",
                sheet="SA-4c / SA-5",
                cell=None,
                snippet=f"SA-4c管理员候选: {', '.join(sorted(sa_admins))} | SA-5执行/审批人候选: {', '.join(sorted(sa5_executors))}",
                basis="常见问题：不同测试项识别的特权用户/执行人未联动分析，可能遗漏特权用户或导致职责分离结论冲突。",
                suggestion="对SA-4c识别的特权用户，与SA-5权限新增/修改执行人、其他底稿中关键操作人做交叉比对，确认是否存在遗漏、共享账号或职责冲突，并统一结论。",
            )
        )

    def _sort_key(item: Finding) -> Tuple[int, str, str]:
        severity_rank = {"高": 0, "中": 1, "低": 2}
        return (severity_rank.get(item.severity, 9), item.sheet, item.issue_type)

    sheets_filter_applied = bool(sheets)
    llm_ac_report: Optional[Dict[str, object]] = None

    llm_ac_report, llm_ac_findings = _llm_check_procedure_pairs(
        wb=wb,
        target_sheets=target_sheets,
        model=str(resolved_model),
        api_key=str(resolved_key),
        base_url=resolved_base_url,
        start_row=int(llm_ac_start_row),
        sleep_seconds=float(llm_ac_sleep_seconds),
    )
    findings.extend(llm_ac_findings)

    by_severity = defaultdict(int)
    by_type = defaultdict(int)
    for item in findings:
        by_severity[item.severity] += 1
        by_type[item.issue_type] += 1

    findings_sorted = sorted(findings, key=_sort_key)

    # ── 证据验证 + 交叉校验 ──
    # 对所有 Finding 运行 _verify_evidence_refs（LLM 发现的 evidence_refs 可能
    # 包含不匹配的 excerpt）和 _cross_validate_finding（确定性规则校验）。
    # Finding 是 frozen dataclass，用 dataclasses.replace 创建修改后的实例。
    for i, f in enumerate(findings_sorted):
        # 1) 验证 evidence_refs excerpt 是否匹配实际单元格文本
        try:
            ev_refs = json.loads(f.evidence_refs) if f.evidence_refs else []
        except Exception:
            ev_refs = []
        if isinstance(ev_refs, list) and ev_refs and f.sheet in wb.sheetnames:
            verified_refs = _verify_evidence_refs(ev_refs, wb[f.sheet])
            if verified_refs != ev_refs:
                findings_sorted[i] = dataclasses.replace(
                    f, evidence_refs=json.dumps(verified_refs, ensure_ascii=False)
                )

        # 2) 交叉验证：标记 needs_review
        f = findings_sorted[i]  # use potentially-updated version
        cross_issues = _cross_validate_finding(f, wb)
        if cross_issues:
            f = dataclasses.replace(f, needs_review=True)
            findings_sorted[i] = f

    # 3) LLM 质疑复核：对 P0 和 needs_review 的 Finding 做一次质疑调用
    challenge_count = 0
    for i, f in enumerate(findings_sorted):
        if f.severity == "P0" or f.needs_review:
            if f.sheet not in wb.sheetnames:
                continue
            ws_for_ctx = wb[f.sheet]
            minimal_ctx = _build_minimal_context(f, ws_for_ctx)
            result = _challenge_finding_with_llm(
                client=client,
                model=str(resolved_model),
                finding=f,
                minimal_context=minimal_ctx,
            )
            if result == "disagree":
                f = dataclasses.replace(f, needs_review=True)
                # 保留原始结论但标记需复核
                findings_sorted[i] = f
                challenge_count += 1
    if challenge_count:
        print(f"LLM质疑复核: {challenge_count} 条发现被质疑标记为需复核", flush=True)

    llm_results: Optional[Dict[int, Dict[str, str]]] = None
    llm_results = _llm_review_findings(
        wb=wb,
        findings_sorted=findings_sorted,
        model=str(resolved_model),
        api_key=str(resolved_key),
        base_url=resolved_base_url,
        batch_size=llm_batch_size,
        sleep_seconds=llm_sleep_seconds,
    )
    if str(output_path).lower().endswith(".xlsx"):
        _write_report_xlsx(
            output_path=output_path,
            started_at=started_at,
            excel_path=excel_path,
            checkpoints_path=checkpoints_path,
            attachments_preview_path=attachments_preview_path,
            attachments_preview=attachments_preview,
            sheet_names=sheet_names,
            target_sheets=target_sheets,
            findings_sorted=findings_sorted,
            by_severity=dict(by_severity),
            by_type=dict(by_type),
            actor_by_sheet=actor_by_sheet,
            sheets_filter_applied=sheets_filter_applied,
            llm_results=llm_results,
            llm_ac_report=llm_ac_report,
            client=client,
            model=str(resolved_model),
        )
    else:
        _write_report_txt(
            output_path=output_path,
            started_at=started_at,
            excel_path=excel_path,
            checkpoints_path=checkpoints_path,
            attachments_preview_path=attachments_preview_path,
            attachments_preview=attachments_preview,
            sheet_names=sheet_names,
            target_sheets=target_sheets,
            findings_sorted=findings_sorted,
            by_severity=dict(by_severity),
            by_type=dict(by_type),
            actor_by_sheet=actor_by_sheet,
            sheets_filter_applied=sheets_filter_applied,
            llm_ac_report=llm_ac_report,
        )

    print("=== 复核完成 ===")
    print(f"报告已生成: {output_path}")
    print(f"问题数: {len(findings_sorted)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="excel_path",
        required=True,
        help="待复核的底稿Excel路径（.xlsx）",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default="excel_analysis_report.xlsx",
        help="输出报告路径（.xlsx 或 .txt），默认：excel_analysis_report.xlsx",
    )
    parser.add_argument(
        "-k",
        "--checkpoints",
        dest="checkpoints_path",
        default=None,
        help="检查要点Excel路径（.xlsx，可选）",
    )
    parser.add_argument(
        "--attachments-preview",
        dest="attachments_preview_path",
        default=None,
        help="附件预览Excel路径（含“图片描述/目录索引”等），用于证据编号/文件匹配与证据-审计步骤一致性检查（可选）",
    )
    parser.add_argument(
        "-s",
        "--sheets",
        dest="sheets",
        default=None,
        help="指定要检查的Sheet（控制点页签），逗号分隔，如：SA-4c,SA-5；不传则检查全部",
    )
    args = parser.parse_args(argv)
    sheets = _parse_sheet_filter(args.sheets)
    generate_report(
        args.excel_path,
        args.output_path,
        args.checkpoints_path,
        attachments_preview_path=args.attachments_preview_path,
        sheets=sheets,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
