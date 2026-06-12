# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

IT审计底稿复核工具集 — reads Excel audit workpapers via `openpyxl`, maps them to a standardized audit schema, and uses LLMs to review workpapers for compliance, methodology, logical consistency, and cross-field consistency. Also includes a tool for reviewing audit finding lists.

**Three self-contained CLI scripts, no shared utility module.** Each script duplicates its own LLM config resolution, cell-text helpers, and `_is_empty()` logic. `STANDARD_SCHEMA_FIELDS` and `FIELD_ALIASES` are defined only in `excel_image_review.py`.

## Commands

```bash
# Install dependencies (use the venv in the repo root)
pip install -r requirements.txt

# Run tests
python -m pytest test_excel_review.py -v
python -m pytest test_excel_image_review_libreoffice.py -v
python -m pytest test_excel_image_review_report_docx.py -v
python -m pytest test_schema_validation.py -v   # schema/validation helpers
python -m pytest test_evidence_chain.py -v      # evidence refs + cross-validation

# Run all tests
python -m pytest test_excel_review.py test_excel_image_review_report_docx.py test_schema_validation.py test_evidence_chain.py -v

# Run a single test
python -m pytest test_excel_review.py::EndToEndStructuredReviewTests::test_full_workflow_generates_structured_payload_and_docx_report -v

# Run scripts
python excel_image_review.py <excel_file> [-o output_dir] [-m model] [-u base_url] [-s sheets] [-n limit]
python analyze_excel.py -i <workpaper.xlsx> [-k checkpoints.xlsx] [-o report.xlsx] [-s sheets] [--attachments-preview preview.xlsx]
python review_audit_findings.py -i <findings.xlsx> -r <reference.xlsx> [-o output.xlsx] [--llm/--no-llm] [--llm-max-items N]
```

`.env` is required for LLM calls (see `.env.example`). Tests use mocked LLM clients and do not need `.env`.

## Architecture

### `excel_image_review.py` — Structured review engine

Class `ExcelImageReviewer` drives the full pipeline: `extract_sheet_structure` (reads cells/merged ranges/detects table regions via BFS) → `map_sheet_to_schema` (maps detected regions to the 10-field standard schema) → `review_structured_sheet` (LLM review, JSON path with Markdown fallback) → `_append_structured_review_to_doc` or `_append_markdown_to_doc` (render to DOCX) → `generate_report`.

`review_structured_sheet` first tries `_try_review_structured_sheet_json` (requests JSON conforming to `_FINDING_RESULT_SCHEMA`); on failure it falls back to `_review_structured_sheet_markdown` (legacy Markdown output, still subject to `_enforce_issue_excerpt_traceability`).  `generate_report` detects dict vs str and dispatches to the matching renderer.

Extraction strategies tried per region, in order:
1. **Tabular** — finds a header row with ≥2 mapped schema fields + anchor fields, then reads data rows below
2. **Step block** — detects "测试步骤" / "标准审计程序" / "执行审计程序" three-column layouts
3. **Key-value** — label→value pairs in adjacent columns
4. **Procedure pair** — full-sheet scan for standard-vs-execution column pairs
5. **Q&A** — fallback: simple question/answer row pairs

Context carry: `control_id`, `control_description`, `audit_objective`, `audit_procedure`, `sample_selection_method` propagate downward through rows (handles merged-cell style layouts). `CONTEXT_CARRY_FIELDS` defines which fields carry forward.

Field aliases for schema mapping are bilingual (Chinese/English) in `FIELD_ALIASES`.

Uses `openai` SDK. Outputs: `<sheet_name>_structured.json` + `review_report.docx`.

### `analyze_excel.py` — ITGC workpaper review (~3200 lines, largest script)

Pipeline: Excel workpaper + optional checkpoints + optional attachments → rule-based checks + LLM checks → deduplicated `Finding` dataclasses → TXT or XLSX report.

Key stages (parallelized with `ThreadPoolExecutor`, up to 4 threads per sheet):
- `_llm_check_sheet_by_checkpoints` — LLM evaluates each checkpoint against sheet content
- `_check_procedure_pairs` + `_llm_check_procedure_pairs` — rule-based + LLM-augmented standard vs executed procedure correspondence
- `_check_attachment_references` — rule-based attachment reference validation
- `_llm_check_evidence_vs_steps` — LLM checks evidence-step consistency
- `_llm_review_findings` — secondary LLM review of aggregated findings
- `_merge_cell_duplicates` — deduplicates findings for the same cell

Output report sheets: 汇总, 问题清单, LLM对应性, LLM调用统计.

Uses `openai` SDK. `Finding` is a `@dataclass(frozen=True)` with fields: `issue_type`, `severity`, `sheet`, `cell`, `snippet`, `basis`, `suggestion`, plus structured-output fields `status` (pass/fail/unknown), `risk_type` (覆盖性/一致性/证据不足/方法性/逻辑性/跨字段一致性), `evidence_refs` (JSON list of `{sheet, cell_or_range, attachment, excerpt}`), `conclusion`, `reasons`, `fix_suggestion_detail`, `unknown_reason`, `needs_review`.  Severity is stored as `P0/P1/P2` internally and mapped to `高/中/低` only at report output.

### `review_audit_findings.py` — Audit findings quality review

Pipeline: Audit findings Excel + reference problem library → rule-based quality flags + similarity matching + optional LLM review → XLSX report.

Uses its own `_LLMClient` (raw `urllib.request`, not the `openai` SDK) with retry logic and alternate endpoint fallback. `FindingRow` and `ReferenceRow` are `@dataclass(frozen=True)` dataclasses.

Key functions:
- `_quality_flags` — rule-based detection (vague descriptions, missing compensating controls, etc.)
- `_similarity` — character bigram Jaccard for matching findings to reference library
- `_keyword_bucket` — classifies findings into categories (access control, change management, etc.)
- `_template_suggestions` — generates template compensating control and risk suggestions by category
- `_derive_suggestions` — combines reference matching + template generation

Output: XLSX with 22-column review suggestions, including structured fields `状态（LLM）`, `严重级别（LLM）`, `风险类型（LLM）`, `结论（LLM）`, `整改建议(结构化)`, `不确定原因`.

## Standard schema

All extraction in `excel_image_review.py` targets a 10-field schema: `control_id`, `control_description`, `audit_objective`, `audit_procedure`, `sample_selection_method`, `sample_size`, `test_steps`, `test_result`, `exception_flag`, `conclusion`.

## Unified LLM finding schema (status / evidence / risk / fix)

All three scripts share an identical `_FINDING_RESULT_SCHEMA` (defined per-script, not shared via import — see "Three self-contained CLI scripts" above).  The schema is the contract every LLM call should respect, and is enforced by `jsonschema` validation plus `_validate_finding_result` / `_repair_finding_result` helpers.

Required fields per finding:
- `status` — `"pass"` / `"fail"` / `"unknown"` (English enum, no Chinese)
- `conclusion` — one-sentence conclusion (≥4 chars)
- `evidence_refs` — array of `{sheet, cell_or_range, attachment?, excerpt?}`; **required when `status == "fail"`**; excerpt must be verbatim from source

Optional but expected for non-pass findings:
- `reasons` — 2-5 bullet points
- `severity` — `P0` / `P1` / `P2` (display-mapped to 高/中/低)
- `risk_type` — one of `覆盖性 / 一致性 / 证据不足 / 方法性 / 逻辑性 / 跨字段一致性`
- `fix_suggestion` — `{missing_field?, supplement_explanation?, required_evidence_type?}`
- `unknown_reason` — required when `status == "unknown"`, ≥10 chars

**Hard rules**:
- `status == "fail"` with empty `evidence_refs` ⇒ downgrade to `unknown`
- `status == "unknown"` without `unknown_reason` ⇒ invalid
- `_repair_finding_result` migrates legacy Chinese values (`无问题 → pass`, `有问题 → fail`, `不确定 → unknown`; `高/中/低 → P0/P1/P2`)
- **`_EXCERPT_MAX_LEN = 2000`** — maximum excerpt length in both `_repair_finding_result` (constructing from fallback fields) and `_verify_evidence_refs` (replacing mismatched excerpts). Must be consistent across all three scripts. Verified by `test_evidence_chain.py::ExcerptMaxLenTests`.
- **`_EXCERPT_CONSTRUCTED_MARKER = "[非逐字原文]"`** — appended to `cell_or_range` when the excerpt was constructed from snippet/basis (not from LLM output or real cell text), helping auditors distinguish evidence sources.

`analyze_excel.py`'s `_llm_request_json_list` accepts `result_schema` + `schema_records` kwargs to validate each result item, repair it if possible, and trigger a retry (with the error text appended to the user prompt) on unrepairable items.

## Hallucination guards

- `_verify_evidence_refs(refs, ws)` — fuzzy-matches each `excerpt` against actual cell text; replaces mismatched excerpts with the real cell text; drops references pointing to empty/invalid cells. Now runs on **all** findings in `generate_report`, not just LLM checkpoints.
- `_cross_validate_finding(finding, wb)` — deterministic rule checks: pass-with-positive-exception-flag, coverage-without-sample-size, excerpt-mismatch, P0-without-evidence. Returns a list of issue codes; findings with issues are marked `needs_review=True`. Now runs on **all** findings in `generate_report` (after `_verify_evidence_refs`).
- `_build_minimal_context(finding, ws, max_chars=2000)` — extracts only the cells explicitly referenced in `evidence_refs`, plus the table header row and adjacent rows.  Reduces LLM input from up to 24,000 chars to 500-2,000 chars, limiting the LLM's surface area for confabulation.
- `_challenge_finding_with_llm` — "质疑 prompt" re-review of P0 / `needs_review` findings; disagreement sets `needs_review=True`. Now wired into `generate_report`: runs after `_cross_validate_finding`, cost-bounded to findings with `severity=="P0"` or `needs_review==True`.

**Pipeline order in `generate_report`**:
1. Per-sheet rule + LLM checks → `findings` list
2. `_llm_check_procedure_pairs` → adds procedure-pair findings
3. Cross-sheet checks (SA-4c/SA-5 disjoint)
4. Sort findings → `findings_sorted`
5. **`_verify_evidence_refs`** on every finding (verifies excerpt against actual cell text)
6. **`_cross_validate_finding`** on every finding (marks `needs_review=True` for rule violations)
7. **`_challenge_finding_with_llm`** on P0/needs_review findings (marks `needs_review=True` on disagreement)
8. `_llm_review_findings` (secondary review, skips LLM-prefixed findings)
9. `_write_report_xlsx` / `_write_report_txt` (includes post-merge `needs_review` check)

## LLM configuration

All scripts read `.env` for API configuration:

- `excel_image_review.py`: only reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- `analyze_excel.py`, `review_audit_findings.py`: env var cascade in priority order — `CHECKER_API_KEY`/`LLM_API_KEY`/`API_KEY`/`DASHSCOPE_API_KEY`/`OPENAI_API_KEY`. Same cascade for base URL (`CHECKER_API_BASE_URL`/`LLM_API_BASE_URL`/`API_BASE_URL`/`OPENAI_BASE_URL`/`BASE_URL`) and model (`CHECKER_MODEL`/`LLM_MODEL`/`MODEL`/`OPENAI_MODEL`). Compatible with any OpenAI-compatible endpoint (DashScope, etc.).
- Optional runtime env vars: `LLM_TIMEOUT` (default 90s), `LLM_EVIDENCE_STEPS_MAX_ITEMS` (default 30, 0 = unlimited).
- `INCLUDE_LLM_REVIEW_COLS` — set to non-empty to add 5 LLM-review columns to the XLSX "问题清单" sheet (analyze_excel.py).

## Commit conventions

Uses conventional commits (Chinese/English): `feat:`, `refactor:`, `chore:`, `fix:`.
