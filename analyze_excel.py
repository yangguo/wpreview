import os
import re
import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import openpyxl
from openpyxl.utils import get_column_letter


@dataclass(frozen=True)
class Finding:
    issue_type: str
    severity: str
    sheet: str
    cell: Optional[str]
    snippet: str
    basis: str
    suggestion: str


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


def _likely_interview_only(execution_text: str) -> bool:
    t = execution_text or ""
    if not any(k in t for k in INTERVIEW_ONLY_KEYWORDS):
        return False
    return not any(k in t for k in EVIDENCE_KEYWORDS)


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
                severity="中",
                sheet=ws_title,
                cell=None,
                snippet="",
                basis="Sheet内未提取到可用于复核的文本单元格。",
                suggestion="确认该Sheet是否为图片/对象或空白；如为图片型底稿需先OCR或改用可读文本版本。",
            )
        ]

    system_prompt = (
        "你是一名严格的IT审计/财务审计质量复核专家。\n"
        "你将收到：\n"
        "1) 某个底稿Sheet的文本化内容（每行含单元格坐标与文字）；\n"
        "2) 该Sheet对应的“检查要点”清单。\n\n"
        "你的任务：逐条检查要点，判断该Sheet是否存在相关问题（未覆盖/证据不足/表述不清/范围不全/仅访谈等）。\n"
        "输出要求：必须输出严格JSON对象：{\"results\": [...]}。\n"
        "results每个元素必须包含字段：\n"
        "- id: 整数\n"
        "- checkpoint: 字符串（原检查要点）\n"
        "- status: \"无问题\"/\"有问题\"/\"不确定\"\n"
        "- severity: \"高\"/\"中\"/\"低\"（当status=无问题时可填\"\"）\n"
        "- issue_type: 字符串（当status=无问题时可填\"\"）\n"
        "- basis: 字符串（简要说明为什么判断有问题/不确定，需引用Sheet内容中的关键句或单元格坐标）\n"
        "- suggestion: 字符串（可执行整改建议，含证据类型/范围/抽样基准等）\n"
        "- related_cells: 字符串数组（尽量给出相关单元格坐标，如\"C15\"；没有则[]）\n"
        "- missing_evidence: 字符串数组（缺失证据类型，如截图/导出清单/日志/台账/审批/协议等；没有则[]）\n"
        "不要输出Markdown代码块，不要输出多余文字。"
    )

    findings: List[Finding] = []
    cell_ref_re = re.compile(r"^[A-Z]{1,3}\d{1,7}$")
    for start in range(0, len(deduped), max(1, int(batch_size))):
        chunk = deduped[start : start + max(1, int(batch_size))]
        end = start + len(chunk)
        print(f"检查要点LLM进度({ws_title}): {start + 1}-{end}/{len(deduped)}", flush=True)
        payload = {
            "sheet": ws_title,
            "checkpoints": [{"id": start + i + 1, "checkpoint": cp} for i, cp in enumerate(chunk)],
            "sheet_text": sheet_text,
        }
        user_prompt = "请按要求逐条复核以下检查要点：\n" + json.dumps(payload, ensure_ascii=False, indent=2)

        last_error: Optional[str] = None
        content = ""
        for attempt in range(1, 4):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                content = resp.choices[0].message.content if resp.choices else ""
                parsed = _try_parse_json(content)
                if isinstance(parsed, dict):
                    parsed = parsed.get("results") or parsed.get("data") or parsed.get("items")
                if not isinstance(parsed, list):
                    raise RuntimeError("LLM返回非JSON results 数组")

                for obj in parsed:
                    if not isinstance(obj, dict):
                        continue
                    status = str(obj.get("status", "")).strip()
                    if status == "无问题":
                        continue
                    checkpoint = str(obj.get("checkpoint", "")).strip()
                    severity = str(obj.get("severity", "")).strip() or ("中" if status == "不确定" else "中")
                    issue_type = str(obj.get("issue_type", "")).strip() or ("检查要点存在问题" if status == "有问题" else "检查要点信息不足/不确定")
                    basis = str(obj.get("basis", "")).strip()
                    suggestion = str(obj.get("suggestion", "")).strip()
                    related_cells = obj.get("related_cells", [])
                    cell = None
                    if isinstance(related_cells, list) and related_cells:
                        for c in related_cells:
                            cc = str(c).strip().upper()
                            if cell_ref_re.match(cc):
                                cell = cc
                                break
                    if not cell and basis:
                        hits = re.findall(r"\b[A-Z]{1,3}\d{1,7}\b", basis.upper())
                        for h in hits:
                            if cell_ref_re.match(h):
                                cell = h
                                break
                    missing_evidence = obj.get("missing_evidence", [])
                    missing_text = ""
                    if isinstance(missing_evidence, list) and missing_evidence:
                        missing_text = "缺失证据: " + "、".join(str(x).strip() for x in missing_evidence if str(x).strip())
                    basis_parts: List[str] = []
                    if checkpoint:
                        basis_parts.append("检查要点: " + checkpoint)
                    if basis:
                        basis_parts.append("依据: " + basis)
                    if missing_text:
                        basis_parts.append(missing_text)
                    basis = "\n".join(p for p in basis_parts if p).strip()

                    snippet = ""
                    if cell:
                        cell_text = _get_cell_value(ws, cell)
                        if cell_text:
                            snippet = _truncate(cell_text, 220)
                    if not snippet and basis:
                        snippet = _truncate(basis.replace("\n", " "), 220)

                    findings.append(
                        Finding(
                            issue_type="LLM判定：检查要点-" + issue_type,
                            severity=severity if severity in {"高", "中", "低"} else "中",
                            sheet=ws_title,
                            cell=cell,
                            snippet=snippet,
                            basis=_truncate(basis or "LLM判定存在问题/不确定", 1200),
                            suggestion=_truncate(suggestion or "对照检查要点补充执行步骤与证据，并在底稿中保留可复核来源。", 1200),
                        )
                    )

                last_error = None
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(min(6, attempt * 2))
                continue

        if last_error:
            findings.append(
                Finding(
                    issue_type="LLM判定：检查要点复核失败",
                    severity="中",
                    sheet=ws_title,
                    cell=None,
                    snippet="",
                    basis=_truncate("检查要点: " + "；".join(chunk) + f"\nLLM调用失败: {last_error}", 1200),
                    suggestion="检查LLM接口配置（.env）、网络连通性；必要时减少检查要点条数或缩短Sheet文本后重试。",
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

    empty_streak = 0
    for row in range(start_row, (ws.max_row or 0) + 1):
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
                            severity="高",
                            sheet=ws_title,
                            cell=c_cell,
                            snippet=_truncate(c_text, 220),
                            basis="执行列内容更像标准模板/判定口径（如“如果/以下/被认为”及大量条款），缺少“我们获取/检查/抽样/复核”等实际执行描述。",
                            suggestion="将该单元格补充为实际执行步骤与获取证据描述（含样本框定方法、样本来源/编号、证据链接/截图/导出）。",
                        )
                    )
                continue

            if _likely_interview_only(c_text):
                findings.append(
                    Finding(
                        issue_type="程序执行不到位/仅依赖访谈",
                        severity="中",
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
                        severity="中",
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
                            severity="中",
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
                            severity="中",
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
                            severity="中",
                            sheet=ws_title,
                            cell=c_cell,
                            snippet=_truncate(c_text, 220),
                            basis="常见问题：调岗应纳入权限变更/禁用控制测试，仅看账号状态不足以覆盖权限调整实质性测试。",
                            suggestion="获取调岗人员名单与用户清单关联，框定调岗且持有账号人员范围，按调岗前后岗位权限差异抽样核查权限变更/禁用证据。",
                        )
                    )

            if any(k in a_text for k in ("密码策略", "密码", "复杂度", "锁定", "时效")):
                if not any(k in c_text for k in ("截图", "参数", "配置", "界面")):
                    findings.append(
                        Finding(
                            issue_type="密码策略证据有效性不足",
                            severity="中",
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
                            severity="中",
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
                            severity="中",
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
                    severity="中",
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
                    severity="中",
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
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
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
        "4) results 内每个元素对应输入的id，且必须包含字段：id, llm_validity(成立/不成立/不确定), llm_severity(高/中/低), llm_comment, llm_missing_evidence, llm_next_actions。\n"
        "5) llm_missing_evidence 与 llm_next_actions 均为字符串数组。\n"
        "6) 不要输出Markdown代码块，不要输出多余解释文字。\n"
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

        last_error = None
        for attempt in range(1, 4):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                content = resp.choices[0].message.content if resp.choices else ""
                parsed = _try_parse_json(content)
                if isinstance(parsed, dict):
                    parsed = parsed.get("results") or parsed.get("data") or parsed.get("items")
                if not isinstance(parsed, list):
                    raise RuntimeError("LLM返回非JSON results 数组")
                for obj in parsed:
                    if not isinstance(obj, dict):
                        continue
                    idx = obj.get("id")
                    if not isinstance(idx, int):
                        continue
                    results[idx] = {
                        "llm_validity": str(obj.get("llm_validity", "")).strip(),
                        "llm_severity": str(obj.get("llm_severity", "")).strip(),
                        "llm_comment": str(obj.get("llm_comment", "")).strip(),
                        "llm_missing_evidence": json.dumps(obj.get("llm_missing_evidence", []), ensure_ascii=False),
                        "llm_next_actions": json.dumps(obj.get("llm_next_actions", []), ensure_ascii=False),
                    }
                last_error = None
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(min(6, attempt * 2))
                continue

        if last_error:
            print(f"LLM复核失败: {start + 1}-{end}/{len(selected)}: {last_error}", flush=True)
            for idx, _ in chunk:
                results[idx] = {
                    "llm_validity": "不确定",
                    "llm_severity": "",
                    "llm_comment": f"LLM调用失败: {last_error}",
                    "llm_missing_evidence": "[]",
                    "llm_next_actions": "[]",
                }
        else:
            print(f"LLM复核完成: {start + 1}-{end}/{len(selected)}", flush=True)

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

    system_prompt = (
        "你是一名严格的审计质量复核专家。\n"
        "你的任务：以A列的【标准审计程序描述】为唯一判断标准，检查C列的【实际执行程序】是否符合标准。\n\n"
        "判断规则：\n"
        "1. 标准描述中要求的审计动作，实际执行中是否包含\n"
        "2. 标准描述中指定的审计对象，实际执行中是否覆盖\n"
        "3. 标准描述中提出的具体条件，实际执行中是否满足\n"
        "4. 标准描述中要求获取的审计证据类型，实际执行中是否获取\n\n"
        "请只回答：【符合】或【不符合】\n"
        "然后换行写【理由：】后面跟简要说明。"
    )
    user_prompt = (
        "【标准审计程序描述 - A列】：\n"
        f"{_clip(standard_text, 900)}\n\n"
        "【实际执行程序 - C列】：\n"
        f"{_clip(execution_text, 900)}\n\n"
        "请判断C列执行程序是否符合A列标准要求。"
    )

    last_error: Optional[str] = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            answer = resp.choices[0].message.content if resp.choices else ""
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

            return True, is_match, reason, answer
        except Exception as e:
            last_error = str(e)
            time.sleep(min(6, attempt * 2))
            continue

    return False, None, last_error or "LLM调用失败", ""


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

    for sheet in target_sheets:
        if sheet not in wb.sheetnames:
            continue
        ws = wb[sheet]
        header_row, standard_col, execution_cols = _detect_layout(ws)
        if not standard_col or not execution_cols:
            continue

        execution_labels = {c: _execution_label(ws, header_row, c) for c in execution_cols}
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

                    if result_label == "✗ 不符合":
                        findings.append(
                            Finding(
                                issue_type="LLM判定：执行程序不符合标准审计程序",
                                severity="高",
                                sheet=ws.title,
                                cell=execution_cell,
                                snippet=_truncate(c_for_judge, 220),
                                basis=_truncate(reason or raw or "LLM判定为不符合", 800),
                                suggestion="补充/修改执行程序以覆盖标准要求的审计动作、对象、条件与证据类型，并在底稿中保留可复核来源（截图/导出清单/日志台账/审批或协议等）。",
                            )
                        )
                    else:
                        findings.append(
                            Finding(
                                issue_type="LLM判定：对应性检查失败/不确定",
                                severity="中",
                                sheet=ws.title,
                                cell=execution_cell,
                                snippet=_truncate(c_for_judge, 220),
                                basis=_truncate(reason or raw or "LLM调用失败或返回不确定", 800),
                                suggestion="检查LLM接口配置（API Key/Base URL/模型名）与网络连通性；必要时缩短单元格内容或重试。",
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
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def _write_report_txt(
    output_path: str,
    started_at: datetime,
    excel_path: str,
    checkpoints_path: Optional[str],
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
    sheet_names: Sequence[str],
    target_sheets: Sequence[str],
    findings_sorted: Sequence[Finding],
    by_severity: Dict[str, int],
    by_type: Dict[str, int],
    actor_by_sheet: Dict[str, Dict[str, List[Tuple[str, str]]]],
    sheets_filter_applied: bool,
    llm_results: Optional[Dict[int, Dict[str, str]]] = None,
    llm_ac_report: Optional[Dict[str, object]] = None,
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    wb.remove(wb.active)

    wrap = Alignment(wrap_text=True, vertical="top")
    header_font = Font(bold=True)

    ws_sum = wb.create_sheet("汇总")
    ws_sum["A1"] = "IT一般控制测试底稿复核报告"
    ws_sum["A1"].font = Font(bold=True, size=14)

    ws_sum["A3"] = "生成时间"
    ws_sum["B3"] = started_at.strftime("%Y-%m-%d %H:%M:%S")
    ws_sum["A4"] = "文件路径"
    ws_sum["B4"] = excel_path
    ws_sum["A5"] = "检查要点"
    ws_sum["B5"] = checkpoints_path or ""
    ws_sum["A6"] = "Sheet数量（全量）"
    ws_sum["B6"] = len(sheet_names)
    ws_sum["A7"] = "Sheet数量（本次检查）"
    ws_sum["B7"] = len(target_sheets)
    ws_sum["A8"] = "本次检查Sheet"
    ws_sum["B8"] = ", ".join(target_sheets) if sheets_filter_applied else ""
    ws_sum["A10"] = "总问题数"
    ws_sum["B10"] = len(findings_sorted)

    ws_sum["A12"] = "严重级别"
    ws_sum["B12"] = "数量"
    ws_sum["A12"].font = header_font
    ws_sum["B12"].font = header_font
    r = 13
    for sev, cnt in sorted(by_severity.items()):
        ws_sum[f"A{r}"] = sev
        ws_sum[f"B{r}"] = cnt
        r += 1

    ws_sum["D12"] = "Top问题类型"
    ws_sum["E12"] = "数量"
    ws_sum["D12"].font = header_font
    ws_sum["E12"].font = header_font
    r = 13
    for issue_type, cnt in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        ws_sum[f"D{r}"] = issue_type
        ws_sum[f"E{r}"] = cnt
        r += 1

    ws_sum.column_dimensions["A"].width = 18
    ws_sum.column_dimensions["B"].width = 110
    ws_sum.column_dimensions["D"].width = 42
    ws_sum.column_dimensions["E"].width = 10
    for cell in ws_sum["B3":"B10"][0]:
        cell.alignment = wrap
    ws_sum["B4"].alignment = wrap
    ws_sum["B5"].alignment = wrap
    ws_sum["B8"].alignment = wrap

    ws_issues = wb.create_sheet("问题清单")
    headers = ["序号", "严重级别", "问题类型", "Sheet", "单元格", "原文摘录", "判定依据", "整改建议"]
    if llm_results is not None:
        headers.extend(["LLM成立性", "LLM严重级别", "LLM复核意见", "LLM缺失证据", "LLM下一步动作"])
    for c, h in enumerate(headers, start=1):
        cell = ws_issues.cell(row=1, column=c, value=h)
        cell.font = header_font
        cell.alignment = Alignment(vertical="top")
    ws_issues.freeze_panes = "A2"

    widths = [8, 10, 28, 14, 10, 70, 60, 60]
    if llm_results is not None:
        widths.extend([12, 12, 70, 50, 50])
    for idx, w in enumerate(widths, start=1):
        ws_issues.column_dimensions[get_column_letter(idx)].width = w

    for idx, item in enumerate(findings_sorted, start=1):
        row = [
            idx,
            item.severity,
            item.issue_type,
            item.sheet,
            item.cell or "-",
            _safe_cell_text(item.snippet),
            _safe_cell_text(item.basis),
            _safe_cell_text(item.suggestion),
        ]
        if llm_results is not None:
            llm = llm_results.get(idx) or {}
            row.extend(
                [
                    llm.get("llm_validity", ""),
                    llm.get("llm_severity", ""),
                    llm.get("llm_comment", ""),
                    llm.get("llm_missing_evidence", ""),
                    llm.get("llm_next_actions", ""),
                ]
            )
        rr = idx + 1
        for cc, value in enumerate(row, start=1):
            cell = ws_issues.cell(row=rr, column=cc, value=value)
            if cc >= 6:
                cell.alignment = wrap
            else:
                cell.alignment = Alignment(vertical="top")

    sev_rows = len(by_severity) if by_severity else 0
    top_type_rows = min(20, len(by_type)) if by_type else 0
    roles_start = max(13 + max(0, sev_rows), 13 + max(0, top_type_rows)) + 3
    ws_sum[f"A{roles_start}"] = "关键角色候选（便于联动复核）"
    ws_sum[f"A{roles_start}"].font = Font(bold=True, size=12)
    ws_sum[f"A{roles_start + 1}"] = "Sheet"
    ws_sum[f"B{roles_start + 1}"] = "管理员候选"
    ws_sum[f"C{roles_start + 1}"] = "执行/审批/复核人候选"
    for col in ("A", "B", "C"):
        ws_sum[f"{col}{roles_start + 1}"].font = header_font
        ws_sum[f"{col}{roles_start + 1}"].alignment = Alignment(vertical="top")
    ws_sum.column_dimensions["C"].width = 70

    r = roles_start + 2
    for sheet in ("SA-4c", "SA-5", "PM-5", "PM-6", "SA-12"):
        if sheet not in actor_by_sheet:
            continue
        admins = sorted({token for _, token in actor_by_sheet[sheet].get("admins", [])})
        executors = sorted({token for _, token in actor_by_sheet[sheet].get("executors", [])})
        if not admins and not executors:
            continue
        ws_sum[f"A{r}"] = sheet
        ws_sum[f"B{r}"] = ", ".join(admins)
        ws_sum[f"C{r}"] = ", ".join(executors)
        ws_sum[f"B{r}"].alignment = wrap
        ws_sum[f"C{r}"].alignment = wrap
        r += 1

    if llm_ac_report:
        ws_llm = wb.create_sheet("LLM对应性")
        ws_llm["A1"] = "LLM对应性检查（标准审计程序 vs 执行审计程序）"
        ws_llm["A1"].font = Font(bold=True, size=14)
        ws_llm["A3"] = "模型"
        ws_llm["B3"] = str(llm_ac_report.get("model", ""))
        ws_llm["A4"] = "接口"
        ws_llm["B4"] = str(llm_ac_report.get("base_url", ""))

        ws_llm["A6"] = "总计检查"
        ws_llm["B6"] = int(llm_ac_report.get("total", 0) or 0)
        ws_llm["A7"] = "符合"
        ws_llm["B7"] = int(llm_ac_report.get("matched", 0) or 0)
        ws_llm["A8"] = "不符合/不确定"
        ws_llm["B8"] = int(llm_ac_report.get("failed", 0) or 0)
        ws_llm["A9"] = "API错误"
        ws_llm["B9"] = int(llm_ac_report.get("api_errors", 0) or 0)
        ws_llm["A10"] = "跳过（执行为空）"
        ws_llm["B10"] = int(llm_ac_report.get("skipped_c_empty", 0) or 0)
        ws_llm["A11"] = "跳过（标准为空）"
        ws_llm["B11"] = int(llm_ac_report.get("skipped_a_empty", 0) or 0)
        ws_llm["A12"] = "跳过（引用/编号）"
        ws_llm["B12"] = int(llm_ac_report.get("skipped_ref", 0) or 0)
        ws_llm["A13"] = "跳过（标题/分类）"
        ws_llm["B13"] = int(llm_ac_report.get("skipped_header", 0) or 0)

        ws_llm["A15"] = "Sheet"
        ws_llm["B15"] = "检查"
        ws_llm["C15"] = "符合"
        ws_llm["D15"] = "不符合/不确定"
        ws_llm["E15"] = "API错误"
        ws_llm["F15"] = "执行空"
        ws_llm["G15"] = "标准空"
        ws_llm["H15"] = "引用"
        ws_llm["I15"] = "标题"
        for col in "ABCDEFGHI":
            ws_llm[f"{col}15"].font = header_font
            ws_llm[f"{col}15"].alignment = Alignment(vertical="top")

        sheet_stats = llm_ac_report.get("sheet_stats") or {}
        r = 16
        if isinstance(sheet_stats, dict):
            for name, st in sheet_stats.items():
                if not isinstance(st, dict):
                    continue
                ws_llm[f"A{r}"] = str(name)
                ws_llm[f"B{r}"] = int(st.get("total", 0) or 0)
                ws_llm[f"C{r}"] = int(st.get("matched", 0) or 0)
                ws_llm[f"D{r}"] = int(st.get("failed", 0) or 0)
                ws_llm[f"E{r}"] = int(st.get("api_errors", 0) or 0)
                ws_llm[f"F{r}"] = int(st.get("skipped_c_empty", 0) or 0)
                ws_llm[f"G{r}"] = int(st.get("skipped_a_empty", 0) or 0)
                ws_llm[f"H{r}"] = int(st.get("skipped_ref", 0) or 0)
                ws_llm[f"I{r}"] = int(st.get("skipped_header", 0) or 0)
                r += 1

        issues_title_row = r + 2
        ws_llm[f"A{issues_title_row}"] = "问题明细（不符合/API错误/不确定）"
        ws_llm[f"A{issues_title_row}"].font = Font(bold=True, size=12)
        headers = ["Sheet", "Row", "标准单元格", "执行单元格", "执行对象", "结果", "理由", "标准审计程序", "执行审计程序", "原始回答"]
        for c, h in enumerate(headers, start=1):
            cell = ws_llm.cell(row=issues_title_row + 1, column=c, value=h)
            cell.font = header_font
            cell.alignment = Alignment(vertical="top")
        widths = [14, 8, 12, 12, 16, 12, 60, 70, 70, 60]
        for idx, w in enumerate(widths, start=1):
            ws_llm.column_dimensions[get_column_letter(idx)].width = w
        ws_llm.column_dimensions["A"].width = 18
        rr = issues_title_row + 2
        rows = llm_ac_report.get("issues")
        if isinstance(rows, list):
            for item in rows:
                if not isinstance(item, dict):
                    continue
                values = [
                    item.get("sheet", ""),
                    item.get("row", ""),
                    item.get("standard_cell", ""),
                    item.get("execution_cell", ""),
                    item.get("execution_label", ""),
                    item.get("result", ""),
                    item.get("reason", ""),
                    item.get("a_text", ""),
                    item.get("c_text", ""),
                    item.get("raw", ""),
                ]
                for cc, v in enumerate(values, start=1):
                    cell = ws_llm.cell(row=rr, column=cc, value=_safe_cell_text(v, 30000))
                    if cc >= 7:
                        cell.alignment = wrap
                    else:
                        cell.alignment = Alignment(vertical="top")
                rr += 1

    wb.save(output_path)


def generate_report(
    excel_path: str,
    output_path: str,
    checkpoints_path: Optional[str] = None,
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

    resolved_key, resolved_base_url, resolved_model = resolve_llm_config()
    if not resolved_key:
        raise RuntimeError(
            "未配置LLM的API Key（请在脚本同目录 .env 中设置 CHECKER_API_KEY/LLM_API_KEY/API_KEY/OPENAI_API_KEY，或设置对应环境变量）"
        )
    client = _ensure_openai_client(api_key=str(resolved_key), base_url=resolved_base_url)

    findings: List[Finding] = []
    actor_by_sheet: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}

    for name in target_sheets:
        ws = wb[name]
        if checkpoints_by_sheet:
            cps = checkpoints_by_sheet.get(name) or checkpoints_by_norm.get(_normalize_sheet_id(name)) or []
            if cps:
                findings.extend(
                    _llm_check_sheet_by_checkpoints(
                        client=client,
                        model=str(resolved_model),
                        ws_title=name,
                        ws=ws,
                        checkpoints=cps,
                        batch_size=6,
                        sleep_seconds=0.2,
                    )
                )
        findings.extend(_check_sheet_scope(name, ws))
        findings.extend(_check_procedure_pairs(name, ws))
        actor_by_sheet[name] = _extract_actor_candidates(ws)

    sa_admins = {token for _, token in actor_by_sheet.get("SA-4c", {}).get("admins", [])}
    sa5_executors = {token for _, token in actor_by_sheet.get("SA-5", {}).get("executors", [])}
    if sa_admins and sa5_executors and sa_admins.isdisjoint(sa5_executors):
        findings.append(
            Finding(
                issue_type="跨测试点特权用户可能遗漏/结论联动不足",
                severity="中",
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
            sheet_names=sheet_names,
            target_sheets=target_sheets,
            findings_sorted=findings_sorted,
            by_severity=dict(by_severity),
            by_type=dict(by_type),
            actor_by_sheet=actor_by_sheet,
            sheets_filter_applied=sheets_filter_applied,
            llm_results=llm_results,
            llm_ac_report=llm_ac_report,
        )
    else:
        _write_report_txt(
            output_path=output_path,
            started_at=started_at,
            excel_path=excel_path,
            checkpoints_path=checkpoints_path,
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
        sheets=sheets,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
