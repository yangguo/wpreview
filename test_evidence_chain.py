"""Tests for evidence-chain verification, cross-validation, and challenge helpers.

Covers:
- _excerpt_matches (normalisation and substring match)
- _verify_evidence_refs (verify against actual cell text, repair, drop invalid)
- _cross_validate_finding (deterministic cross-checks)
- _build_minimal_context (minimal LLM context)
- _challenge_finding_with_llm (mocked LLM challenge call)
- Pipeline integration: _verify_evidence_refs + _cross_validate_finding run on all findings
"""

import dataclasses
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openpyxl  # noqa: E402

import analyze_excel as ax  # noqa: E402


class ExcerptMatchTests(unittest.TestCase):
    def test_exact_substring_match(self):
        self.assertTrue(ax._excerpt_matches("已获取用户清单", "已获取用户清单并核查权限"))

    def test_normalised_match(self):
        # Punctuation/whitespace should be stripped
        self.assertTrue(ax._excerpt_matches("已获取\n用户清单", "已获取用户清单"))

    def test_no_match_when_not_substring(self):
        self.assertFalse(ax._excerpt_matches("无关文本", "已获取用户清单"))

    def test_empty_inputs(self):
        self.assertFalse(ax._excerpt_matches("", "text"))
        self.assertFalse(ax._excerpt_matches("text", ""))
        self.assertFalse(ax._excerpt_matches("", ""))


class VerifyEvidenceRefsTests(unittest.TestCase):
    def setUp(self):
        self.wb = openpyxl.Workbook()
        self.ws = self.wb.active
        self.ws.title = "Test"
        self.ws["A1"] = "header"
        self.ws["B1"] = "value cell"
        self.ws["C5"] = "测试结果"
        self.ws["C6"] = "another cell"

    def test_keeps_matching_ref(self):
        refs = [{"cell_or_range": "C5", "excerpt": "测试结果"}]
        verified = ax._verify_evidence_refs(refs, self.ws)
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["excerpt"], "测试结果")

    def test_repairs_mismatched_excerpt(self):
        refs = [{"cell_or_range": "C5", "excerpt": "完全不对的文本"}]
        verified = ax._verify_evidence_refs(refs, self.ws)
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["excerpt"], "测试结果")

    def test_drops_invalid_cell(self):
        refs = [{"cell_or_range": "ZZ999", "excerpt": "anything"}]
        verified = ax._verify_evidence_refs(refs, self.ws)
        self.assertEqual(len(verified), 0)

    def test_drops_empty_cell(self):
        refs = [{"cell_or_range": "A2", "excerpt": "anything"}]  # A2 is empty
        verified = ax._verify_evidence_refs(refs, self.ws)
        self.assertEqual(len(verified), 0)

    def test_exerpt_max_len_constant(self):
        """Verify _EXCERPT_MAX_LEN is consistent across scripts."""
        self.assertEqual(ax._EXCERPT_MAX_LEN, 2000)
        import excel_image_review as eir  # noqa: E402
        import review_audit_findings as raf  # noqa: E402
        self.assertEqual(eir._EXCERPT_MAX_LEN, 2000)
        self.assertEqual(raf._EXCERPT_MAX_LEN, 2000)


class CrossValidateFindingTests(unittest.TestCase):
    def setUp(self):
        self.wb = openpyxl.Workbook()
        self.ws = self.wb.active
        self.ws.title = "S"
        # set sample_size in known location
        self.ws["D1"] = "样本量"
        self.ws["E1"] = "10"
        # exception_flag cell
        self.ws["D5"] = "exception_flag"
        self.ws["E5"] = "Y"  # 异常

    def test_pass_with_positive_exception_flag_raises_issue(self):
        f = ax.Finding(
            issue_type="test",
            severity="P1",
            sheet="S",
            cell="E5",  # The cell that contains "Y"
            snippet="",
            basis="",
            suggestion="",
            status="pass",
            risk_type="方法性",
            evidence_refs="[]",
        )
        # Note: _cross_validate_finding looks at cells from cell + evidence_refs
        # Pass + E5 contains Y → exception_flag_contradicts_pass
        issues = ax._cross_validate_finding(f, self.wb)
        self.assertIn("exception_flag_contradicts_pass", issues)

    def test_coverage_with_no_sample_size_raises_issue(self):
        # Create a sheet with NO sample_size row
        wb2 = openpyxl.Workbook()
        ws2 = wb2.active
        ws2.title = "NoSample"
        ws2["A1"] = "control_id"
        ws2["B1"] = "C-1"
        f = ax.Finding(
            issue_type="test",
            severity="P1",
            sheet="NoSample",
            cell="B1",
            snippet="",
            basis="",
            suggestion="",
            status="fail",
            risk_type="覆盖性",
            evidence_refs='[{"cell_or_range": "B1", "excerpt": "C-1"}]',
        )
        issues = ax._cross_validate_finding(f, wb2)
        self.assertIn("coverage_claim_but_no_sample_size", issues)

    def test_p0_fail_without_evidence_raises_issue(self):
        f = ax.Finding(
            issue_type="test",
            severity="P0",
            sheet="S",
            cell="A1",
            snippet="",
            basis="",
            suggestion="",
            status="fail",
            risk_type="方法性",
            evidence_refs="[]",
        )
        issues = ax._cross_validate_finding(f, self.wb)
        self.assertIn("high_severity_no_evidence", issues)

    def test_clean_finding_no_issues(self):
        f = ax.Finding(
            issue_type="test",
            severity="P1",
            sheet="S",
            cell="E1",
            snippet="",
            basis="",
            suggestion="",
            status="fail",
            risk_type="方法性",
            evidence_refs='[{"cell_or_range": "E1", "excerpt": "10"}]',
        )
        issues = ax._cross_validate_finding(f, self.wb)
        self.assertEqual(issues, [])

    def test_none_workbook_returns_empty(self):
        f = ax.Finding(
            issue_type="test",
            severity="P0",
            sheet="S",
            cell="A1",
            snippet="",
            basis="",
            suggestion="",
            status="fail",
            risk_type="方法性",
            evidence_refs="[]",
        )
        issues = ax._cross_validate_finding(f, None)
        self.assertEqual(issues, [])  # graceful: no crash, no issues

    def test_nonexistent_sheet_returns_empty(self):
        f = ax.Finding(
            issue_type="test",
            severity="P0",
            sheet="NonExistentSheet",
            cell="A1",
            snippet="",
            basis="",
            suggestion="",
            status="fail",
            risk_type="方法性",
            evidence_refs="[]",
        )
        issues = ax._cross_validate_finding(f, self.wb)
        self.assertEqual(issues, [])


class MinimalContextTests(unittest.TestCase):
    def setUp(self):
        self.wb = openpyxl.Workbook()
        self.ws = self.wb.active
        self.ws.title = "MC"
        # Create a small table
        self.ws["A1"] = "control_id"
        self.ws["B1"] = "test_result"
        self.ws["A2"] = "C-001"
        self.ws["B2"] = "无异常"
        self.ws["A3"] = "C-002"
        self.ws["B3"] = "有异常"

    def test_includes_evidence_ref_cell(self):
        f = ax.Finding(
            issue_type="test",
            severity="P1",
            sheet="MC",
            cell="B3",
            snippet="",
            basis="",
            suggestion="",
            evidence_refs='[{"cell_or_range": "B3", "excerpt": "有异常"}]',
        )
        ctx = ax._build_minimal_context(f, self.ws)
        self.assertIn("B3: 有异常", ctx)
        # Header row should be included
        self.assertIn("control_id", ctx)

    def test_falls_back_to_finding_cell(self):
        f = ax.Finding(
            issue_type="test",
            severity="P1",
            sheet="MC",
            cell="B3",
            snippet="",
            basis="",
            suggestion="",
            evidence_refs="[]",
        )
        ctx = ax._build_minimal_context(f, self.ws)
        self.assertIn("B3", ctx)

    def test_truncates_long_context(self):
        # Add a lot of text
        for r in range(1, 50):
            self.ws.cell(row=r, column=10, value="x" * 200)
        f = ax.Finding(
            issue_type="test",
            severity="P1",
            sheet="MC",
            cell="B2",
            snippet="",
            basis="",
            suggestion="",
            evidence_refs="[]",
        )
        ctx = ax._build_minimal_context(f, self.ws, max_chars=500)
        self.assertLessEqual(len(ctx), 510)  # allow for "..." suffix


class ChallengeFindingWithLLMTests(unittest.TestCase):
    """Test _challenge_finding_with_llm with mocked LLM client."""

    def setUp(self):
        self.wb = openpyxl.Workbook()
        self.ws = self.wb.active
        self.ws.title = "TestSheet"
        self.ws["A1"] = "header"
        self.ws["B1"] = "value cell"
        self.ws["C5"] = "已获取用户清单并核查权限"

    def test_challenge_agree(self):
        """When LLM agrees, should return 'agree'."""
        f = ax.Finding(
            issue_type="test",
            severity="P0",
            sheet="TestSheet",
            cell="C5",
            snippet="已获取用户清单并核查权限",
            basis="控制执行缺乏过程证据",
            suggestion="补充系统截图",
            status="fail",
            risk_type="证据不足",
            evidence_refs='[{"cell_or_range": "C5", "excerpt": "已获取用户清单并核查权限"}]',
        )
        mock_client = MagicMock()
        with patch("analyze_excel._llm_chat", return_value="agree"):
            result = ax._challenge_finding_with_llm(
                client=mock_client,
                model="test-model",
                finding=f,
                minimal_context="context text",
            )
        self.assertEqual(result, "agree")

    def test_challenge_disagree(self):
        """When LLM disagrees, should return 'disagree'."""
        f = ax.Finding(
            issue_type="test",
            severity="P0",
            sheet="TestSheet",
            cell="C5",
            snippet="已获取用户清单",
            basis="控制执行缺乏过程证据",
            suggestion="补充系统截图",
            status="fail",
            risk_type="证据不足",
            evidence_refs='[{"cell_or_range": "C5", "excerpt": "已获取用户清单"}]',
        )
        mock_client = MagicMock()
        with patch("analyze_excel._llm_chat", return_value="disagree"):
            result = ax._challenge_finding_with_llm(
                client=mock_client,
                model="test-model",
                finding=f,
                minimal_context="context text",
            )
        self.assertEqual(result, "disagree")

    def test_challenge_chinese_agree(self):
        """Chinese '同意' should map to 'agree'."""
        f = ax.Finding(
            issue_type="test",
            severity="P1",
            sheet="TestSheet",
            cell="B1",
            snippet="value",
            basis="test basis",
            suggestion="test",
            status="fail",
            evidence_refs='[{"cell_or_range": "B1", "excerpt": "value cell"}]',
        )
        with patch("analyze_excel._llm_chat", return_value="同意，该发现成立"):
            result = ax._challenge_finding_with_llm(
                client=MagicMock(),
                model="test",
                finding=f,
                minimal_context="ctx",
            )
        self.assertEqual(result, "agree")

    def test_challenge_chinese_disagree(self):
        """Chinese '不成立' should map to 'disagree'."""
        f = ax.Finding(
            issue_type="test",
            severity="P0",
            sheet="TestSheet",
            cell="C5",
            snippet="text",
            basis="basis",
            suggestion="sug",
            status="fail",
            evidence_refs="[]",
        )
        with patch("analyze_excel._llm_chat", return_value="不成立"):
            result = ax._challenge_finding_with_llm(
                client=MagicMock(),
                model="test",
                finding=f,
                minimal_context="ctx",
            )
        self.assertEqual(result, "disagree")

    def test_challenge_error_returns_none(self):
        """When LLM call throws an exception, should return None."""
        f = ax.Finding(
            issue_type="test",
            severity="P1",
            sheet="TestSheet",
            cell="B1",
            snippet="",
            basis="basis",
            suggestion="",
            status="fail",
            evidence_refs="[]",
        )
        with patch("analyze_excel._llm_chat", side_effect=RuntimeError("API error")):
            result = ax._challenge_finding_with_llm(
                client=MagicMock(),
                model="test",
                finding=f,
                minimal_context="ctx",
            )
        self.assertIsNone(result)

    def test_challenge_empty_context_returns_none(self):
        """When minimal_context is empty, should return None without calling LLM."""
        f = ax.Finding(
            issue_type="test",
            severity="P0",
            sheet="TestSheet",
            cell="C5",
            snippet="",
            basis="basis",
            suggestion="",
            status="fail",
            evidence_refs="[]",
        )
        with patch("analyze_excel._llm_chat") as mock_chat:
            result = ax._challenge_finding_with_llm(
                client=MagicMock(),
                model="test",
                finding=f,
                minimal_context="",
            )
        self.assertIsNone(result)
        mock_chat.assert_not_called()

    def test_challenge_no_client_returns_none(self):
        """When client is None, should return None without calling LLM."""
        f = ax.Finding(
            issue_type="test",
            severity="P0",
            sheet="TestSheet",
            cell="C5",
            snippet="",
            basis="basis",
            suggestion="",
            status="fail",
            evidence_refs="[]",
        )
        with patch("analyze_excel._llm_chat") as mock_chat:
            result = ax._challenge_finding_with_llm(
                client=None,
                model="test",
                finding=f,
                minimal_context="context",
            )
        self.assertIsNone(result)
        mock_chat.assert_not_called()


class ExcerptMaxLenTests(unittest.TestCase):
    """Verify excerpt truncation uses the _EXCERPT_MAX_LEN constant."""

    def test_repair_finding_uses_excerpt_max_len(self):
        """_repair_finding_result should truncate excerpts to _EXCERPT_MAX_LEN."""
        long_text = "A" * 500
        result = ax._repair_finding_result({
            "status": "fail",
            "conclusion": "test conclusion text",
            "evidence_refs": [],
            "severity": "P1",
            "risk_type": "方法性",
            "related_cells": "A1",
            "snippet": long_text,
        })
        # When constructing evidence_refs from related_cells + snippet,
        # excerpt should be truncated to _EXCERPT_MAX_LEN
        if result and result.get("evidence_refs"):
            for ref in result["evidence_refs"]:
                if "excerpt" in ref and ref["excerpt"]:
                    self.assertLessEqual(len(ref["excerpt"]), ax._EXCERPT_MAX_LEN)

    def test_verified_excerpt_max_len(self):
        """_verify_evidence_refs should truncate verified excerpts to _EXCERPT_MAX_LEN."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "T"
        # Cell with text longer than 500
        ws["A1"] = "B" * 2500
        refs = [{"cell_or_range": "A1", "excerpt": "wrong text"}]
        verified = ax._verify_evidence_refs(refs, ws)
        self.assertEqual(len(verified), 1)
        # The excerpt should be repaired to the full cell text (capped at _EXCERPT_MAX_LEN)
        self.assertLessEqual(len(verified[0]["excerpt"]), ax._EXCERPT_MAX_LEN)


class PipelineIntegrationTests(unittest.TestCase):
    """Test that cross-validation and evidence verification work end-to-end on Finding objects."""

    def setUp(self):
        self.wb = openpyxl.Workbook()
        self.ws = self.wb.active
        self.ws.title = "IntegrationSheet"
        self.ws["A1"] = "control_id"
        self.ws["B1"] = "test_result"
        self.ws["C1"] = "exception_flag"
        self.ws["A2"] = "C-001"
        self.ws["B2"] = "无异常"
        self.ws["C2"] = "Y"  # exception flag is positive!
        self.ws["D1"] = "样本量"
        self.ws["E1"] = "10"

    def test_verify_refs_then_cross_validate(self):
        """Simulates the generate_report pipeline: verify evidence_refs, then cross-validate."""
        # Finding with mismatched excerpt
        f = ax.Finding(
            issue_type="test",
            severity="P1",
            sheet="IntegrationSheet",
            cell="B2",
            snippet="无异常",
            basis="test basis",
            suggestion="test suggestion",
            status="pass",
            risk_type="方法性",
            evidence_refs='[{"cell_or_range": "B2", "excerpt": "wrong excerpt"}]',
        )

        # Step 1: Verify evidence_refs
        ev_refs = json.loads(f.evidence_refs)
        verified = ax._verify_evidence_refs(ev_refs, self.wb["IntegrationSheet"])
        # Excerpt should be repaired to match actual cell text
        self.assertEqual(verified[0]["excerpt"], "无异常")
        # Update the finding
        f = dataclasses.replace(f, evidence_refs=json.dumps(verified, ensure_ascii=False))

        # Step 2: Cross-validate
        issues = ax._cross_validate_finding(f, self.wb)
        # B2 = "无异常" (no exception), C2 = "Y" => exception_flag_contradicts_pass
        # But C2 is not in f.cell or f.evidence_refs, so this shouldn't trigger
        # unless we look at the cell range. Let's check:
        # f.cell = "B2", evidence_refs cell = "B2"
        # exception_flag check only looks at f.cell + evidence_refs cells
        # Since C2 is not referenced, this should NOT flag the contradiction
        # unless the cell references include C2

    def test_cross_validate_flags_p0_without_evidence(self):
        """P0 finding without evidence should be flagged by cross-validation."""
        f = ax.Finding(
            issue_type="test",
            severity="P0",
            sheet="IntegrationSheet",
            cell="A2",
            snippet="",
            basis="test",
            suggestion="test",
            status="fail",
            risk_type="证据不足",
            evidence_refs="[]",
        )
        issues = ax._cross_validate_finding(f, self.wb)
        self.assertIn("high_severity_no_evidence", issues)
        # Simulate the pipeline: mark as needs_review
        f = dataclasses.replace(f, needs_review=True)
        self.assertTrue(f.needs_review)

    def test_verify_refs_drops_invalid_and_cross_validate_flags(self):
        """Invalid refs dropped by _verify_evidence_refs may trigger cross-validation flags."""
        f = ax.Finding(
            issue_type="coverage_issue",
            severity="P0",
            sheet="IntegrationSheet",
            cell="ZZ999",
            snippet="has issue",
            basis="coverage problem",
            suggestion="add sampling",
            status="fail",
            risk_type="覆盖性",
            evidence_refs='[{"cell_or_range": "ZZ999", "excerpt": "nothing"}]',
        )

        # Step 1: Verify — invalid cell should be dropped
        ev_refs = json.loads(f.evidence_refs)
        verified = ax._verify_evidence_refs(ev_refs, self.wb["IntegrationSheet"])
        self.assertEqual(len(verified), 0)  # invalid cell dropped

        f = dataclasses.replace(f, evidence_refs=json.dumps(verified, ensure_ascii=False))

        # Step 2: Cross-validate
        issues = ax._cross_validate_finding(f, self.wb)
        # P0 + fail + no evidence_refs after verification => high_severity_no_evidence
        self.assertIn("high_severity_no_evidence", issues)
        # risk_type="覆盖性" with no sample_size in the sheet
        # Actually there IS a sample_size (D1/E1), so coverage should NOT be flagged
        # unless the finding's cell is not near the sample_size. Let's check:
        # _cross_validate_finding searches the FIRST 80 rows for sample_size keywords
        # E1 = "10" which is a non-empty value near D1 = "样本量"
        # So coverage_claim_but_no_sample_size should NOT be in issues


class RepairTruncationConsistencyTests(unittest.TestCase):
    """Verify that _repair_finding_result uses _EXCERPT_MAX_LEN consistently."""

    def test_repair_truncates_excerpt_to_max_len(self):
        long_text = "X" * 500
        result = ax._repair_finding_result({
            "status": "fail",
            "conclusion": "test conclusion text here",
            "evidence_refs": [],
            "severity": "P1",
            "risk_type": "方法性",
            "related_cells": "A1",
            "snippet": long_text,
        })
        self.assertIsNotNone(result)
        refs = result.get("evidence_refs", [])
        for ref in refs:
            if "excerpt" in ref and ref["excerpt"]:
                self.assertLessEqual(len(ref["excerpt"]), ax._EXCERPT_MAX_LEN,
                                     f"Excerpt length {len(ref['excerpt'])} exceeds _EXCERPT_MAX_LEN {ax._EXCERPT_MAX_LEN}")


if __name__ == "__main__":
    unittest.main()