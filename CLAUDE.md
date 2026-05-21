# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

IT审计底稿复核工具集 — reads Excel audit workpapers via `openpyxl`, maps them to a standardized audit schema, and uses LLMs to review workpapers for compliance, methodology, logical consistency, and cross-field consistency. Also includes supporting tools for reviewing audit finding lists and checking procedure correspondence.

## Commands

```bash
# Install dependencies (use the venv in the repo root)
pip install -r requirements.txt

# Run tests
python -m pytest test_excel_review.py -v
python -m pytest test_excel_image_review_libreoffice.py -v
python -m pytest test_excel_image_review_report_docx.py -v

# Run a single test
python -m pytest test_excel_review.py::EndToEndStructuredReviewTests::test_full_workflow_generates_structured_payload_and_docx_report -v
```

`.env` is required for LLM calls (see `.env.example`). Tests use mocked LLM clients and do not need `.env`.

## Architecture

Four independent scripts, each a self-contained CLI tool:

### `excel_image_review.py` — Structured review engine

Core pipeline: `extract_sheet_structure` (reads cells/merged ranges/detects table regions via BFS) → `map_sheet_to_schema` (maps detected regions to the 10-field standard schema using one of 4 extraction strategies) → `review_structured_sheet` (LLM review) → `generate_report` (DOCX).

Extraction strategies tried per region, in order:
1. **Tabular** — finds a header row with ≥2 mapped schema fields + anchor fields, then reads data rows below
2. **Step block** — detects "测试步骤" / "标准审计程序" / "执行审计程序" three-column layouts
3. **Key-value** — label→value pairs in adjacent columns
4. **Procedure pair** — full-sheet scan for standard-vs-execution column pairs
5. **Q&A** — fallback: simple question/answer row pairs

Context carry: `control_id`, `control_description`, `audit_objective`, `audit_procedure`, `sample_selection_method` propagate downward through rows (handles merged-cell style layouts).

Field aliases for schema mapping are bilingual (Chinese/English) in `FIELD_ALIASES`.

### `analyze_excel.py` — ITGC workpaper review

LLM-based review with checkpoint-driven evaluation, correspondence checking between standard and executed audit procedures, and secondary review of identified issues. Outputs multi-sheet XLSX reports (汇总, 问题清单, LLM对应性, LLM调用统计). Supports optional attachment evidence preview.

### `review_audit_findings.py` — Audit findings quality review

Reads two Excel inputs: an audit findings list and a reference problem library. For each finding, runs rule-based quality flags (`_quality_flags`), computes similarity (character bigram Jaccard) against the reference library, derives suggested compensating controls and risk descriptions by keyword-bucket templating, and optionally calls LLM for deeper review. Output: XLSX with 16-column review suggestions.

### `底稿复核设计部份.py` — Procedure correspondence checker

Checks whether executed audit procedures (C列) match standard audit procedures (A列) row-by-row using LLM judgment. Supports single-row and batch modes across multiple sheets. Auto-detects column layout by scanning header rows.

## Standard schema

All extraction targets a 10-field schema: `control_id`, `control_description`, `audit_objective`, `audit_procedure`, `sample_selection_method`, `sample_size`, `test_steps`, `test_result`, `exception_flag`, `conclusion`.

## LLM configuration

All scripts read `.env` for API configuration:

- `excel_image_review.py`: only reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- `analyze_excel.py`, `review_audit_findings.py`, `底稿复核设计部份.py`: env var cascade in priority order — `CHECKER_API_KEY`/`LLM_API_KEY`/`API_KEY`/`DASHSCOPE_API_KEY`/`OPENAI_API_KEY`. Same cascade for base URL (`CHECKER_API_BASE_URL`/`LLM_API_BASE_URL`/`API_BASE_URL`/`OPENAI_BASE_URL`/`BASE_URL`) and model (`CHECKER_MODEL`/`LLM_MODEL`/`MODEL`/`OPENAI_MODEL`). Compatible with any OpenAI-compatible endpoint (DashScope, etc.).

## Commit conventions

Uses conventional commits (Chinese/English): `feat:`, `refactor:`, `chore:`, `fix:`.
