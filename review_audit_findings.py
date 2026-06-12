#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jsonschema
import openpyxl


@dataclass(frozen=True)
class FindingRow:
    row_index: int
    defect_id: str
    category: str
    control_type: str
    application: str
    issue_desc: str
    compensating_control: str
    risk_impact: str


@dataclass(frozen=True)
class ReferenceRow:
    row_index: int
    control_point: str
    application: str
    issue_desc: str
    compensating_control: str
    risk_impact: str


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
    if status == "fail":
        refs = obj.get("evidence_refs") or []
        if not isinstance(refs, list) or len(refs) == 0:
            errors.append("status=fail but evidence_refs is empty")
    if status == "unknown":
        reason = str(obj.get("unknown_reason", "")).strip()
        if len(reason) < 10:
            errors.append("status=unknown but unknown_reason is empty or <10 chars")
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

    old_status = str(repaired.get("status", "")).strip()
    if old_status == "无问题":
        repaired["status"] = "pass"
    elif old_status == "有问题":
        repaired["status"] = "fail"
    elif old_status == "不确定":
        repaired["status"] = "unknown"

    status = repaired.get("status", "fail")

    sev = str(repaired.get("severity", "")).strip()
    if sev in _SEVERITY_FROM_CHINESE:
        repaired["severity"] = _SEVERITY_FROM_CHINESE[sev]
    elif sev not in ("P0", "P1", "P2", ""):
        repaired["severity"] = "P1"
    if status != "pass" and not repaired.get("severity"):
        repaired["severity"] = "P1"

    if not repaired.get("conclusion"):
        basis = str(repaired.get("basis", "")).strip()
        if basis:
            repaired["conclusion"] = basis[:200]
        else:
            repaired["conclusion"] = f"发现{status}类问题"

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

    if status != "pass" and not repaired.get("risk_type"):
        repaired["risk_type"] = "证据不足"

    if status == "unknown":
        reason = str(repaired.get("unknown_reason", "")).strip()
        if len(reason) < 10:
            repaired["unknown_reason"] = "LLM未说明不确定原因：需要补充更多信息以判定"

    if not repaired.get("reasons"):
        basis = str(repaired.get("basis", "")).strip()
        if basis:
            repaired["reasons"] = [basis[:300]]
        else:
            repaired["reasons"] = [repaired.get("conclusion", "")]

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


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def _norm(s: Optional[str]) -> str:
    if s is None:
        return ""
    return _WS.sub(" ", str(s).strip())


def _norm_compact(s: Optional[str]) -> str:
    t = _norm(s).lower()
    t = _PUNCT.sub("", t)
    return t


def _cell_text(ws, row: int, col: int) -> str:
    v = ws.cell(row=row, column=col).value
    return _norm(v)


def _scan_header_row(ws, required_headers: Sequence[str], max_scan_rows: int = 40, max_scan_cols: int = 60) -> Tuple[int, Dict[str, int]]:
    best_row = 0
    best_hit = 0
    best_map: Dict[str, int] = {}
    max_r = min(ws.max_row or 0, int(max_scan_rows))
    max_c = min(ws.max_column or 0, int(max_scan_cols))
    for r in range(1, max_r + 1):
        header_map: Dict[str, int] = {}
        for c in range(1, max_c + 1):
            v = _cell_text(ws, r, c)
            if not v:
                continue
            header_map[v] = c
        hit = sum(1 for h in required_headers if any(h in k for k in header_map.keys()))
        if hit > best_hit:
            best_hit = hit
            best_row = r
            best_map = header_map
        if best_hit == len(required_headers):
            break
    col_map: Dict[str, int] = {}
    for h in required_headers:
        for k, c in best_map.items():
            if h in k and h not in col_map:
                col_map[h] = c
                break
    return best_row, col_map


def _read_findings(findings_path: Path) -> Tuple[str, List[FindingRow]]:
    wb = openpyxl.load_workbook(str(findings_path), data_only=True)
    ws = wb[wb.sheetnames[0]]
    required = ("缺陷编号", "类别", "控制类型", "涉及应用程序", "问题描述", "补偿性控制", "风险及影响")
    header_row, col_map = _scan_header_row(ws, required_headers=required)
    if not header_row or any(h not in col_map for h in required):
        raise RuntimeError(f"无法识别审计发现清单表头。sheet={ws.title} header_row={header_row} col_map={col_map}")

    end_row = ws.max_row or 0
    marker_phrases = ("上年度/上次审计发现整改情况", "上年度", "上次审计", "以往审计", "历史审计")
    max_scan_col = min(ws.max_column or 0, 25)
    for r in range(header_row + 1, (ws.max_row or 0) + 1):
        row_text = " ".join(_cell_text(ws, r, c) for c in range(1, max_scan_col + 1) if _cell_text(ws, r, c))
        if not row_text:
            continue
        if any(p in row_text for p in marker_phrases) and ("整改" in row_text or "整改情况" in row_text):
            end_row = r - 1
            break

    rows: List[FindingRow] = []
    for r in range(header_row + 1, max(header_row + 1, end_row) + 1):
        defect_id = _cell_text(ws, r, col_map["缺陷编号"])
        if not defect_id:
            continue
        if any(k in defect_id for k in ("上年度", "上次审计", "以往审计", "历史审计", "整改情况", "整改")) and "发现" in defect_id:
            continue
        if any(k in defect_id for k in ("缺陷编号", "类别", "控制类型", "涉及应用程序", "问题描述", "补偿性控制", "风险及影响")):
            continue
        issue_desc = _strip_prior_audit_remediation(_cell_text(ws, r, col_map["问题描述"]))
        comp = _strip_prior_audit_remediation(_cell_text(ws, r, col_map["补偿性控制"]))
        risk = _strip_prior_audit_remediation(_cell_text(ws, r, col_map["风险及影响"]))
        row = FindingRow(
            row_index=r,
            defect_id=defect_id,
            category=_cell_text(ws, r, col_map["类别"]),
            control_type=_cell_text(ws, r, col_map["控制类型"]),
            application=_cell_text(ws, r, col_map["涉及应用程序"]),
            issue_desc=issue_desc,
            compensating_control=comp,
            risk_impact=risk,
        )
        rows.append(row)
    return ws.title, rows


def _read_reference(reference_path: Path) -> Tuple[str, List[ReferenceRow]]:
    wb = openpyxl.load_workbook(str(reference_path), data_only=True)
    ws = wb[wb.sheetnames[0]]
    required = ("控制点", "涉及应用程序", "问题描述", "补偿性控制", "风险及影响")
    header_row, col_map = _scan_header_row(ws, required_headers=required)
    if not header_row or any(h not in col_map for h in required):
        raise RuntimeError(f"无法识别参考问题库表头。sheet={ws.title} header_row={header_row} col_map={col_map}")

    rows: List[ReferenceRow] = []
    for r in range(header_row + 1, (ws.max_row or 0) + 1):
        issue_desc = _cell_text(ws, r, col_map["问题描述"])
        if not issue_desc:
            continue
        rows.append(
            ReferenceRow(
                row_index=r,
                control_point=_cell_text(ws, r, col_map["控制点"]),
                application=_cell_text(ws, r, col_map["涉及应用程序"]),
                issue_desc=issue_desc,
                compensating_control=_cell_text(ws, r, col_map["补偿性控制"]),
                risk_impact=_cell_text(ws, r, col_map["风险及影响"]),
            )
        )
    return ws.title, rows


def _char_bigrams(s: str) -> List[str]:
    t = _norm_compact(s)
    if len(t) <= 1:
        return [t] if t else []
    return [t[i : i + 2] for i in range(len(t) - 1)]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(x for x in a if x)
    sb = set(x for x in b if x)
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _similarity(a: str, b: str) -> float:
    na = _norm_compact(a)
    nb = _norm_compact(b)
    if not na or not nb:
        return 0.0
    ja = _jaccard(_char_bigrams(na), _char_bigrams(nb))
    if na == nb:
        return 1.0
    prefix = 1.0 if (na[:8] and nb[:8] and (na[:8] == nb[:8])) else 0.0
    return min(1.0, 0.75 * ja + 0.25 * prefix)


def _keyword_bucket(text: str) -> str:
    t = _norm(text)
    if not t:
        return "other"
    buckets: List[Tuple[str, Tuple[str, ...]]] = [
        ("security", ("病毒", "防病毒", "杀毒", "木马", "恶意", "勒索", "防火墙", "入侵", "漏洞", "补丁", "端点", "EDR", "AV")),
        ("incident", ("问题事件", "事件", "故障", "工单", "响应", "处置", "升级", "总结", "根因", "复盘")),
        ("envsep", ("测试环境", "生产环境", "隔离", "环境隔离", "测试记录", "测试证据", "UAT", "SIT")),
        ("change", ("变更", "发布", "上线", "回退", "版本", "代码", "配置变更", "变更审批", "紧急变更")),
        ("access", ("账号", "账户", "权限", "角色", "授权", "注销", "离职", "超权", "sa", "root", "新增用户")),
        ("job", ("批处理", "作业", "调度", "定时任务", "任务", "scheduler", "job")),
        ("log", ("日志", "监控", "告警", "审计日志", "追踪", "留痕", "日志保留")),
        ("backup", ("备份", "恢复", "容灾", "演练", "rto", "rpo")),
        ("interface", ("接口", "对账", "传输", "批量导入", "文件传输", "sftp", "ftp")),
    ]
    scores: Dict[str, int] = {}
    for name, kws in buckets:
        scores[name] = sum(1 for k in kws if k and k in t)
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else "other"


def _quality_flags(issue_desc: str, comp: str, risk: str) -> List[str]:
    flags: List[str] = []
    idesc = _norm(issue_desc)
    c = _norm(comp)
    r = _norm(risk)

    if len(_norm_compact(idesc)) < 18:
        flags.append("问题描述过短/不具体")
    if not any(k in idesc for k in ("未", "缺少", "缺失", "未能", "未按", "未留痕", "不符合", "未见")):
        if len(_norm_compact(idesc)) < 45:
            flags.append("问题描述缺少明确现状/差异点")
    if any(k in idesc for k in ("可能存在", "存在风险")) and len(_norm_compact(idesc)) < 60:
        flags.append("问题描述偏笼统")

    if not c:
        flags.append("缺少补偿性控制描述")
    else:
        if any(k in c for k in ("附加审计程序", "审计人员", "审计程序")) and not any(k in c for k in ("管理层", "业务", "运维", "信息安全", "控制", "机制")):
            flags.append("补偿性控制疑似写成审计程序/审计人员执行步骤")
        if any(k in c for k in ("有效", "已有效", "充分")) and not any(k in c for k in ("证据", "记录", "截图", "导出", "日志", "台账", "审批", "抽样")):
            flags.append("补偿性控制结论化但缺少证据/机制描述")
        if not any(k in c for k in ("频率", "每", "按", "定期", "每日", "每周", "每月", "季度", "年度")):
            flags.append("补偿性控制缺少频率/周期")
        if not any(k in c for k in ("责任", "负责人", "岗位", "部门", "复核", "审批", "监督")):
            flags.append("补偿性控制缺少责任主体/复核机制")

    if not r:
        flags.append("缺少风险及影响描述")
    else:
        if "可能" not in r and "导致" not in r and "风险" in r and len(_norm_compact(r)) < 50:
            flags.append("风险及影响偏口号化")
        if "财务" not in r and any(k in idesc for k in ("报表", "凭证", "记账", "会计", "金额", "收入", "成本", "费用", "资产", "负债")):
            flags.append("风险及影响未连接到财务报表/错报后果")
        if any(k in r for k in ("影响较小", "无影响")) and any(k in idesc for k in ("未", "缺少", "缺失", "不符合")):
            flags.append("风险及影响与问题严重性可能不匹配")

    if c and any(k in c for k in ("已有效", "有效运行", "未见异常")) and any(k in idesc for k in ("未执行", "未留痕", "缺失", "未按")):
        flags.append("补偿性控制与问题描述存在潜在矛盾")

    return flags


def _template_suggestions(bucket: str, application: str) -> Tuple[str, str]:
    app = application or "相关系统"
    if bucket == "security":
        comp = (
            f"对{app}相关终端/服务器统一部署并集中管理防病毒/终端防护策略（如禁止用户自行退出/修改策略），"
            "定期检查病毒库/引擎更新与策略生效情况并留存巡检记录；"
            "对异常告警建立处置流程与闭环记录，并对关键安全设备（防火墙/EDR）日志进行抽样复核。"
        )
        risk = (
            "若终端防护与安全策略未统一管理，可能导致恶意代码入侵、横向移动或关键系统被勒索/破坏，"
            "引发业务中断、数据泄露或关键财务数据被篡改等风险，进而带来合规风险与财务报表错报风险。"
        )
        return comp, risk
    if bucket == "incident":
        comp = (
            f"对{app}建立问题事件（Incident/Problem）管理流程：登记受理、分级响应、处置与关闭标准，"
            "留存工单、根因分析与复盘记录；对重大/重复事件由管理层或独立岗位定期复核处置及时性与有效性，"
            "并推动预防性改进形成闭环。"
        )
        risk = (
            "若问题事件管理缺失，可能导致故障/异常无法及时响应与根因消除，重复事件增加业务中断与数据处理错误风险，"
            "并削弱对关键流程异常的侦测能力，进而影响财务数据完整性、准确性与合规性。"
        )
        return comp, risk
    if bucket == "envsep":
        comp = (
            f"对{app}建立生产/测试环境隔离与访问控制机制（含账号权限、网络/数据库隔离），"
            "明确变更迁移至生产前的测试要求与留痕（测试用例、测试结果、缺陷关闭、UAT签字等），"
            "并由独立岗位对发布前测试证据与迁移记录进行复核，留存审批与发布记录。"
        )
        risk = (
            "若生产与测试环境未有效隔离或测试记录不完整，可能导致生产数据/程序被不当访问或未充分测试的变更进入生产，"
            "引发系统功能异常、数据处理错误或关键参数被不当修改，进而影响财务数据完整性与准确性，造成财务报表错报风险。"
        )
        return comp, risk
    if bucket == "access":
        comp = (
            f"由{app}系统所有者/信息安全负责人与业务负责人对关键账号与高风险权限（如管理员、超权角色）进行定期复核，"
            "基于系统导出权限清单与人员在岗/离职信息核对授权合理性；对新增/变更/注销授权留存审批与执行记录，"
            "对异常授权及时纠正并形成整改闭环。"
        )
        risk = (
            f"若权限管理不规范，可能导致未经授权访问或超权操作，进而引发关键交易/主数据被篡改、关键配置被修改、"
            "不当入账或对账失败等风险，造成财务报表重大错报风险及合规/舞弊风险。"
        )
        return comp, risk
    if bucket == "change":
        comp = (
            f"对{app}的变更实施变更审批、发布前测试验证与发布后回溯检查；"
            "对紧急变更设置事后补批与独立复核机制，留存变更单、测试证据、发布记录与回退预案/执行记录。"
        )
        risk = (
            "若变更控制不足，可能导致未经授权或未充分测试的变更进入生产环境，引发系统功能异常、接口数据错误、"
            "关键参数被不当修改，进而导致交易处理错误或会计记录不准确，造成财务报表错报风险与业务中断风险。"
        )
        return comp, risk
    if bucket == "job":
        comp = (
            f"对{app}关键批处理/作业调度建立清单化管理，设置运行监控与失败告警，"
            "并由运维/业务定期复核作业运行日志与异常处理记录；对关键作业参数与调度配置变更实施审批与留痕。"
        )
        risk = (
            "若作业调度与监控不足，可能导致关键批处理未按期运行、运行失败未及时发现或错误处理，"
            "进而造成数据处理不完整、对账差异或交易遗漏，影响财务数据的完整性与准确性。"
        )
        return comp, risk
    if bucket == "log":
        comp = (
            f"对{app}关键操作与异常事件启用审计日志并设置集中留存与保留期限；"
            "由安全/运维定期复核日志与告警处置记录，对高风险操作进行抽样核查并形成闭环。"
        )
        risk = (
            "若日志留痕与监控不足，可能导致异常操作无法及时发现或事后追溯困难，"
            "增加未授权变更/舞弊行为未被识别的风险，进而带来合规风险与财务报表错报风险。"
        )
        return comp, risk
    if bucket == "backup":
        comp = (
            f"对{app}关键数据与配置执行定期备份并加密/隔离存放，定期开展恢复演练并留存演练记录；"
            "对备份失败设置告警并形成异常处置闭环。"
        )
        risk = (
            "若备份与恢复控制不足，可能导致系统故障或数据损坏时无法及时恢复，造成业务中断或数据丢失，"
            "进而影响关键交易处理与财务数据完整性，带来财务报表错报及持续经营相关风险。"
        )
        return comp, risk
    if bucket == "interface":
        comp = (
            f"对{app}接口/数据传输建立对账与异常处理机制，定期核对传输文件数量/金额、接口调用结果与差异，"
            "并留存对账、差异分析与整改闭环记录；对接口参数变更实施审批与测试。"
        )
        risk = (
            "若接口传输与对账不足，可能导致数据传输不完整或错误未被及时发现，造成上下游数据不一致、"
            "关键交易遗漏或金额错误，影响财务数据准确性与完整性，带来财务报表错报风险。"
        )
        return comp, risk
    comp = (
        f"针对{app}相关控制缺陷，建议建立可执行的补偿性控制：明确控制活动、责任人、频率、覆盖范围与留存证据，"
        "对发现的异常及时纠正并形成闭环。"
    )
    risk = (
        "若控制未能有效执行，可能导致关键流程缺乏必要的预防/侦测控制，增加错误处理、未经授权操作或舞弊未被识别的风险，"
        "进而影响财务数据的准确性、完整性与合规性。"
    )
    return comp, risk


def _pick_reference(issue_desc: str, reference_rows: Sequence[ReferenceRow]) -> Tuple[Optional[ReferenceRow], float]:
    best: Optional[ReferenceRow] = None
    best_score = 0.0
    for rr in reference_rows:
        score = _similarity(issue_desc, rr.issue_desc)
        if score > best_score:
            best = rr
            best_score = score
    return best, float(best_score)


_CTRL_POINT_RE = re.compile(r"\b(?:SA|PM|NS|PE)[-_ ]?\d{1,2}[A-Za-z]?\b", re.IGNORECASE)


def _extract_control_points(text: str) -> List[str]:
    t = _norm(text)
    if not t:
        return []
    items: List[str] = []
    for m in _CTRL_POINT_RE.findall(t):
        s = m.strip().replace("_", "-").replace(" ", "-")
        s = re.sub(r"-{2,}", "-", s)
        s = s.upper()
        if s and s not in items:
            items.append(s)
    return items


def _control_point_match_rows(control_points: Sequence[str], ref_rows: Sequence[ReferenceRow]) -> List[ReferenceRow]:
    cps = set(cp.strip().upper() for cp in control_points if cp)
    if not cps:
        return []
    matched: List[ReferenceRow] = []
    for rr in ref_rows:
        cp = rr.control_point.strip().upper() if rr.control_point else ""
        if cp and cp in cps:
            matched.append(rr)
    return matched


def _derive_suggestions(row: FindingRow, ref_rows: Sequence[ReferenceRow]) -> Tuple[str, str, Optional[ReferenceRow], float]:
    bucket = _keyword_bucket(" ".join([row.control_type, row.issue_desc, row.compensating_control, row.risk_impact]))
    control_points = _extract_control_points(row.application)
    scoped_refs = _control_point_match_rows(control_points, ref_rows)
    ref, score = _pick_reference(row.issue_desc, scoped_refs or ref_rows)

    def _looks_like_audit_step(text: str) -> bool:
        t = _norm(text)
        if not t:
            return False
        return any(k in t for k in ("IT审计", "审计组", "审计人员", "审计人员通过", "进一步审计", "我们检查", "访谈", "获取并检查"))

    if ref and ref.compensating_control and ref.risk_impact and (scoped_refs or score >= 0.22):
        comp_s = ref.compensating_control
        risk_s = ref.risk_impact
        if row.application and row.application not in ("-", "—") and row.application not in comp_s:
            comp_s = comp_s.replace("相关系统", row.application)
        if _looks_like_audit_step(comp_s):
            comp_s, risk_s = _template_suggestions(bucket, row.application)
        return comp_s, risk_s, ref, score
    comp_s, risk_s = _template_suggestions(bucket, row.application)
    return comp_s, risk_s, ref, score


def _write_report(
    *,
    output_path: Path,
    findings_sheet: str,
    findings_rows: Sequence[FindingRow],
    reference_sheet: str,
    reference_rows: Sequence[ReferenceRow],
    llm_enabled: bool,
    llm_max_items: int,
) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "审阅建议"

    headers = [
        "缺陷编号",
        "类别",
        "控制类型",
        "涉及应用程序",
        "问题描述",
        "补偿性控制及有效性（原）",
        "风险及影响（原）",
        "问题点（规则）",
        "建议补偿性控制",
        "建议风险及影响",
        "参考库匹配控制点",
        "参考库相似度",
        "状态（LLM）",
        "严重级别（LLM）",
        "风险类型（LLM）",
        "结论（LLM）",
        "问题点（LLM）",
        "建议补偿性控制（LLM）",
        "建议风险及影响（LLM）",
        "整改建议(结构化)",
        "不确定原因",
        "来源定位",
    ]
    ws.append(headers)

    llm_client = _LLMClient.from_env() if llm_enabled else None
    if llm_enabled and not llm_client:
        print("提示：已启用LLM审阅，但未检测到API Key/Base URL 配置（将仅输出规则/参考库建议）。")
    if llm_client:
        print(f"LLM已启用：model={llm_client.model} endpoint={llm_client.endpoint}")
    for idx, fr in enumerate(findings_rows, start=1):
        flags = _quality_flags(fr.issue_desc, fr.compensating_control, fr.risk_impact)
        comp_s, risk_s, ref, score = _derive_suggestions(fr, reference_rows)
        llm_issues = ""
        llm_comp = ""
        llm_risk = ""
        llm_status = ""
        llm_severity = ""
        llm_risk_type = ""
        llm_conclusion = ""
        llm_fix = ""
        llm_unknown = ""
        if llm_client and (llm_max_items <= 0 or idx <= llm_max_items):
            llm_payload = _llm_review_one(fr, reference_rows, llm_client)
            llm_issues = _norm(llm_payload.get("issues") or "")
            llm_comp = _norm(llm_payload.get("improved_compensating_control") or "")
            llm_risk = _norm(llm_payload.get("improved_risk_impact") or "")
            llm_status = _norm(str(llm_payload.get("status") or ""))
            raw_sev = _norm(str(llm_payload.get("severity") or ""))
            llm_severity = _SEVERITY_DISPLAY.get(raw_sev, raw_sev)
            llm_risk_type = _norm(str(llm_payload.get("risk_type") or ""))
            llm_conclusion = _norm(str(llm_payload.get("conclusion") or ""))
            llm_unknown = _norm(str(llm_payload.get("unknown_reason") or ""))
            fix_obj = llm_payload.get("fix_suggestion") or {}
            if isinstance(fix_obj, dict):
                fix_parts = []
                if fix_obj.get("missing_field"):
                    fix_parts.append(f"缺: {str(fix_obj['missing_field'])[:200]}")
                if fix_obj.get("supplement_explanation"):
                    fix_parts.append(f"补: {str(fix_obj['supplement_explanation'])[:200]}")
                if fix_obj.get("required_evidence_type"):
                    fix_parts.append(f"需证据: {str(fix_obj['required_evidence_type'])[:200]}")
                llm_fix = " | ".join(fix_parts)
        source = f"{findings_sheet}!R{fr.row_index}"
        ws.append(
            [
                fr.defect_id,
                fr.category,
                fr.control_type,
                fr.application,
                fr.issue_desc,
                fr.compensating_control,
                fr.risk_impact,
                "；".join(flags) if flags else "",
                comp_s,
                risk_s,
                ref.control_point if ref else "",
                round(float(score), 3),
                llm_status,
                llm_severity,
                llm_risk_type,
                llm_conclusion,
                llm_issues,
                llm_comp,
                llm_risk,
                llm_fix,
                llm_unknown,
                source,
            ]
        )

    ws2 = wb.create_sheet("摘要")
    ws2.append(["输入审计发现清单", str(findings_sheet)])
    ws2.append(["输入参考问题库", str(reference_sheet)])
    ws2.append(["审计发现条数", len(findings_rows)])
    ws2.append(["参考条数", len(reference_rows)])

    for col in range(1, len(headers) + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 22
    ws.column_dimensions["E"].width = 44
    ws.column_dimensions["F"].width = 44
    ws.column_dimensions["G"].width = 44
    ws.column_dimensions["H"].width = 36
    ws.column_dimensions["I"].width = 44
    ws.column_dimensions["J"].width = 44
    ws.column_dimensions["M"].width = 12
    ws.column_dimensions["N"].width = 12
    ws.column_dimensions["O"].width = 16
    ws.column_dimensions["P"].width = 36
    ws.column_dimensions["Q"].width = 36
    ws.column_dimensions["R"].width = 44
    ws.column_dimensions["S"].width = 44
    ws.column_dimensions["T"].width = 50
    ws.column_dimensions["U"].width = 30
    ws.column_dimensions["V"].width = 28

    wb.save(str(output_path))


_PRIOR_REMEDIATION_LINE_RE = re.compile(r"^(?:上年度|上次审计|上期|以往审计|历史审计|去年).{0,30}整改.{0,80}$")
_PRIOR_REMEDIATION_INLINE_RE = re.compile(r"(?:上年度|上次审计|上期|以往审计|历史审计|去年).{0,30}整改.{0,200}")


def _strip_prior_audit_remediation(text: str) -> str:
    t = _norm(text)
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines()]
    kept = [ln for ln in lines if ln and not _PRIOR_REMEDIATION_LINE_RE.match(ln)]
    t2 = "\n".join(kept) if kept else t
    t2 = _PRIOR_REMEDIATION_INLINE_RE.sub("", t2).strip()
    return _norm(t2)


def _load_dotenv_map() -> Dict[str, str]:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return {}
    env_map: Dict[str, str] = {}
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            raw = raw_line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            k, v = raw.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                env_map[k] = v
    except Exception:
        return {}
    return env_map


def _resolve_llm_endpoint_and_key() -> Tuple[Optional[str], Optional[str], str]:
    env_map = _load_dotenv_map()
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
    )
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
        or "gpt-4o"
    )
    if base_url:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/chat/completions"):
            if base_url.endswith("/v1"):
                base_url = base_url + "/chat/completions"
            elif base_url.endswith("/api/v3"):
                base_url = base_url + "/chat/completions"
            else:
                base_url = base_url + "/v1/chat/completions"
    return api_key.strip() if api_key else None, base_url.strip() if base_url else None, str(model).strip()


class _LLMClient:
    def __init__(self, *, api_key: str, endpoint: str, model: str, timeout_seconds: int = 120, max_tokens: int = 700):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = int(timeout_seconds)
        self.max_tokens = int(max_tokens)

    @classmethod
    def from_env(cls) -> Optional["_LLMClient"]:
        api_key, endpoint, model = _resolve_llm_endpoint_and_key()
        if not api_key or not endpoint:
            return None
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "700") or "700")
        timeout = int(os.getenv("LLM_TIMEOUT", "120") or "120")
        return cls(api_key=api_key, endpoint=endpoint, model=model, timeout_seconds=timeout, max_tokens=max_tokens)

    def chat_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.1, retries: int = 2) -> Dict[str, object]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": float(temperature),
            "max_tokens": int(self.max_tokens),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Optional[str] = None
        endpoints = [self.endpoint] + _alternate_llm_endpoints(self.endpoint)
        for attempt in range(1, max(1, int(retries)) + 2):
            try:
                for ep in endpoints:
                    req = urllib.request.Request(
                        ep,
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    try:
                        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                            raw = resp.read().decode("utf-8", errors="replace")
                    except urllib.error.HTTPError as he:
                        if int(getattr(he, "code", 0) or 0) == 404 and ep != endpoints[-1]:
                            last_error = f"HTTP Error 404: Not Found ({ep})"
                            continue
                        raise
                    parsed = json.loads(raw)
                    content = ""
                    try:
                        content = parsed["choices"][0]["message"]["content"]
                    except Exception:
                        content = ""
                    data = _try_parse_json_obj(content)
                    if isinstance(data, dict):
                        return data
                    raise RuntimeError("LLM返回非JSON对象")
            except Exception as e:
                last_error = str(e)
                if attempt < max(1, int(retries)) + 1:
                    time.sleep(min(8.0, 1.5 * attempt))
                    continue
                return {"issues": f"LLM调用失败：{last_error}"}


def _alternate_llm_endpoints(endpoint: str) -> List[str]:
    ep = (endpoint or "").strip()
    if not ep:
        return []
    out: List[str] = []
    if "/v1/chat/completions" in ep:
        out.append(ep.replace("/v1/chat/completions", "/chat/completions"))
    if "/chat/completions" in ep and "/v1/chat/completions" not in ep:
        out.append(ep.replace("/chat/completions", "/v1/chat/completions"))
    if ep.endswith("/v1") and not ep.endswith("/v1/chat/completions"):
        out.append(ep.rstrip("/") + "/chat/completions")
    if ep.endswith("/v1/") and not ep.endswith("/v1/chat/completions"):
        out.append(ep.rstrip("/") + "/chat/completions")
    return [x for x in out if x and x != endpoint]


def _try_parse_json_obj(text: str) -> Optional[object]:
    s = _norm(text)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _clip_text(text: str, limit: int) -> str:
    s = _norm(text)
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def _reference_hints(row: FindingRow, ref_rows: Sequence[ReferenceRow], max_items: int = 3) -> List[ReferenceRow]:
    cps = _extract_control_points(row.application)
    scoped = _control_point_match_rows(cps, ref_rows)
    candidates = scoped or list(ref_rows)
    ranked = sorted(((rr, _similarity(row.issue_desc, rr.issue_desc)) for rr in candidates), key=lambda x: x[1], reverse=True)
    return [rr for rr, _ in ranked[: max(0, int(max_items))] if rr and (rr.compensating_control or rr.risk_impact)]


def _llm_review_one(row: FindingRow, ref_rows: Sequence[ReferenceRow], client: _LLMClient) -> Dict[str, object]:
    hints = _reference_hints(row, ref_rows, max_items=3)
    hints_text = "\n".join(
        [
            f"- 控制点：{h.control_point}\n  问题描述：{_clip_text(h.issue_desc, 220)}\n  补偿性控制：{_clip_text(h.compensating_control, 220)}\n  风险及影响：{_clip_text(h.risk_impact, 220)}"
            for h in hints
        ]
    )
    system_prompt = (
        "你是一名资深IT审计质量复核专家。你要审阅一条审计发现是否逻辑一致、表述是否专业，并给出更好的补偿性控制与风险及影响描述。"
        "要求：输出必须是严格JSON对象；用专业中文；不要输出多余文本。"
    )
    user_prompt = (
        f"【审计发现】\n"
        f"- 缺陷编号：{row.defect_id}\n"
        f"- 类别：{row.category}\n"
        f"- 控制类型：{row.control_type}\n"
        f"- 涉及应用程序/控制点：{row.application}\n"
        f"- 问题描述：{_clip_text(row.issue_desc, 900)}\n"
        f"- 补偿性控制及有效性（原）：{_clip_text(row.compensating_control, 900)}\n"
        f"- 风险及影响（原）：{_clip_text(row.risk_impact, 900)}\n\n"
        f"【参考问题库片段】\n{hints_text or '(无)'}\n\n"
        "请输出严格JSON对象，字段如下：\n"
        '{\n'
        '  "status": "pass"/"fail"/"unknown",\n'
        '  "conclusion": "一句话结论",\n'
        '  "reasons": ["要点1", "要点2"],\n'
        '  "evidence_refs": [{"sheet": "...", "cell_or_range": "...", "excerpt": "..."}], \n'  # may be empty for this script
        '  "severity": "P0"/"P1"/"P2",\n'
        '  "risk_type": "覆盖性"/"一致性"/"证据不足"/"方法性"/"逻辑性"/"跨字段一致性",\n'
        '  "issues": "指出逻辑不一致/不专业点，尽量具体（可为空字符串）",\n'
        '  "improved_compensating_control": "改写后的补偿性控制及有效性",\n'
        '  "improved_risk_impact": "改写后的风险及影响",\n'
        '  "fix_suggestion": {"missing_field": "...", "supplement_explanation": "...", "required_evidence_type": "..."},\n'
        '  "unknown_reason": "（status=unknown时必填，≥10字符）"\n'
        '}\n'
    )
    raw = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
    if not isinstance(raw, dict):
        return raw
    # 验证 + 修复
    valid, needs_retry = _validate_llm_results([raw], schema_records=None)
    if valid:
        return valid[0]
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="审阅审计发现清单：一致性/专业性检查 + 补偿性控制与风险描述建议（参考问题库）")
    parser.add_argument("-i", "--input", required=True, help="审计发现汇总表Excel路径（.xlsx）")
    parser.add_argument("-r", "--reference", required=True, help="参考问题库Excel路径（.xlsx）")
    parser.add_argument("-o", "--output", default="", help="输出报告路径（.xlsx），默认在输入文件同目录生成")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--llm", dest="llm", action="store_true", help="启用大模型二次审阅（默认启用）")
    llm_group.add_argument("--no-llm", dest="llm", action="store_false", help="关闭大模型二次审阅")
    parser.set_defaults(llm=True)
    parser.add_argument("--llm-max-items", type=int, default=0, help="限制调用LLM的条数（0表示不限制）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = Path(args.input).expanduser().resolve()
    ref_path = Path(args.reference).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    if not ref_path.exists():
        raise FileNotFoundError(str(ref_path))

    out = args.output.strip()
    if out:
        output_path = Path(out).expanduser().resolve()
    else:
        output_path = input_path.with_name(input_path.stem + "_审阅建议.xlsx")

    findings_sheet, findings_rows = _read_findings(input_path)
    reference_sheet, reference_rows = _read_reference(ref_path)
    _write_report(
        output_path=output_path,
        findings_sheet=findings_sheet,
        findings_rows=findings_rows,
        reference_sheet=reference_sheet,
        reference_rows=reference_rows,
        llm_enabled=bool(args.llm),
        llm_max_items=int(args.llm_max_items or 0),
    )
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

