"""Tests for unified finding JSON schema and validation/repair helpers.

Covers the schema/validation helpers added in Phase 1–3 of the optimisation:
- _validate_finding_result
- _repair_finding_result
- _validate_llm_results
- fail→unknown downgrade when no evidence_refs
- status migration (中文 → pass/fail/unknown)
- severity migration (高/中/低 → P0/P1/P2)
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_excel as ax  # noqa: E402
import excel_image_review as eir  # noqa: E402
import review_audit_findings as raf  # noqa: E402


VALID_FINDING = {
    "status": "fail",
    "conclusion": "控制执行无过程证据佐证",
    "evidence_refs": [{"cell_or_range": "C15", "excerpt": "已获取用户清单并核查权限"}],
    "severity": "P0",
    "risk_type": "证据不足",
    "reasons": ["未提供系统截图", "仅访谈记录"],
    "fix_suggestion": {"required_evidence_type": "系统截图或导出清单"},
}


class SchemaValidationTests(unittest.TestCase):
    """Verify _validate_finding_result accepts/rejects the right inputs."""

    def test_valid_finding_passes(self):
        ok, errs = ax._validate_finding_result(VALID_FINDING)
        self.assertTrue(ok, errs)
        self.assertEqual(errs, [])

    def test_missing_conclusion_fails(self):
        bad = dict(VALID_FINDING)
        bad["conclusion"] = ""
        ok, errs = ax._validate_finding_result(bad)
        self.assertFalse(ok)
        # Either JSON Schema's minLength error or our additional check
        self.assertTrue(any("conclusion" in e or "too short" in e for e in errs))

    def test_fail_without_evidence_refs_fails(self):
        bad = dict(VALID_FINDING)
        bad["evidence_refs"] = []
        ok, errs = ax._validate_finding_result(bad)
        self.assertFalse(ok)
        self.assertTrue(any("evidence_refs" in e for e in errs))

    def test_unknown_without_unknown_reason_fails(self):
        bad = dict(VALID_FINDING)
        bad["status"] = "unknown"
        bad["evidence_refs"] = []
        bad["unknown_reason"] = ""
        ok, errs = ax._validate_finding_result(bad)
        self.assertFalse(ok)
        self.assertTrue(any("unknown_reason" in e for e in errs))


class SchemaRepairTests(unittest.TestCase):
    """Verify _repair_finding_result migrates legacy LLM outputs."""

    def test_status_migration_chinese(self):
        bad = {
            "status": "有问题",
            "conclusion": "覆盖不全描述",
            "evidence_refs": [{"cell_or_range": "A1", "excerpt": "x"}],
            "severity": "高",
        }
        repaired = ax._repair_finding_result(bad)
        self.assertEqual(repaired["status"], "fail")
        self.assertEqual(repaired["severity"], "P0")

    def test_severity_migration(self):
        for cn, p in [("高", "P0"), ("中", "P1"), ("低", "P2")]:
            repaired = ax._repair_finding_result({
                "status": "fail",
                "conclusion": "x" * 10,
                "severity": cn,
                "evidence_refs": [{"cell_or_range": "A1", "excerpt": "y"}],
            })
            self.assertEqual(repaired["severity"], p, f"failed for {cn}")

    def test_fail_with_no_evidence_downgrades_to_unknown(self):
        bad = {
            "status": "fail",
            "conclusion": "问题描述有证据",
            "evidence_refs": [],
            "severity": "P1",
            "risk_type": "方法性",
        }
        repaired = ax._repair_finding_result(bad)
        self.assertEqual(repaired["status"], "unknown")
        self.assertIn("无法引用原始证据", repaired["unknown_reason"])
        self.assertEqual(repaired["severity"], "P2")

    def test_unknown_reason_auto_generated(self):
        bad = {
            "status": "unknown",
            "conclusion": "信息不足需要补充",
            "evidence_refs": [],
        }
        repaired = ax._repair_finding_result(bad)
        self.assertGreaterEqual(len(repaired["unknown_reason"]), 10)

    def test_repair_succeeds_for_valid_finding(self):
        repaired = ax._repair_finding_result(VALID_FINDING)
        ok, _ = ax._validate_finding_result(repaired)
        self.assertTrue(ok)


class BatchValidationTests(unittest.TestCase):
    """Verify _validate_llm_results handles mixed-validity batches."""

    def test_batch_with_mixed_items(self):
        # The middle item has conclusion too short to repair → it gets dropped
        # (not downgraded to unknown, because unknown also requires a valid conclusion)
        results = [
            VALID_FINDING,
            {"status": "fail", "conclusion": "问题描述有内容", "evidence_refs": []},  # downgrades to unknown
            {"status": "有问题", "conclusion": "覆盖不全描述", "evidence_refs": [{"cell_or_range": "A1", "excerpt": "x"}], "severity": "高"},  # passes after repair
        ]
        valid, needs_retry = ax._validate_llm_results(results)
        # all 3 should be valid after repair
        self.assertEqual(len(valid), 3)
        # The second item should have been downgraded to unknown
        self.assertEqual(valid[1]["status"], "unknown")
        # The third item should have migrated to fail/P0
        self.assertEqual(valid[2]["status"], "fail")
        self.assertEqual(valid[2]["severity"], "P0")


class CrossScriptSchemaParityTests(unittest.TestCase):
    """Verify all three scripts have equivalent schema/validation helpers."""

    def test_all_have_schema(self):
        for mod in (ax, eir, raf):
            self.assertTrue(hasattr(mod, "_FINDING_RESULT_SCHEMA"))
            self.assertIn("status", mod._FINDING_RESULT_SCHEMA["properties"])

    def test_all_have_validators(self):
        for mod in (ax, eir, raf):
            for name in ("_validate_finding_result", "_repair_finding_result", "_validate_llm_results"):
                self.assertTrue(callable(getattr(mod, name, None)), f"{name} missing in {mod.__name__}")

    def test_severity_mapping_consistent(self):
        for mod in (ax, eir, raf):
            self.assertEqual(mod._SEVERITY_FROM_CHINESE["高"], "P0")
            self.assertEqual(mod._SEVERITY_FROM_CHINESE["中"], "P1")
            self.assertEqual(mod._SEVERITY_FROM_CHINESE["低"], "P2")


class FindingDataclassTests(unittest.TestCase):
    """Verify Finding dataclass has new fields and backward compatibility."""

    def test_finding_with_only_legacy_fields(self):
        # 7-arg form: original signature
        f = ax.Finding("issue", "中", "Sheet1", "A1", "snippet", "basis", "sug")
        self.assertEqual(f.status, "fail")
        self.assertEqual(f.evidence_refs, "[]")
        self.assertEqual(f.risk_type, "")

    def test_finding_with_new_fields(self):
        f = ax.Finding(
            issue_type="issue",
            severity="P0",
            sheet="Sheet1",
            cell="A1",
            snippet="snippet",
            basis="basis",
            suggestion="sug",
            status="fail",
            risk_type="覆盖性",
            evidence_refs='[{"cell_or_range": "A1"}]',
            conclusion="问题",
            reasons='["reason1"]',
            fix_suggestion_detail='{"required_evidence_type": "截图"}',
            unknown_reason="",
            needs_review=False,
        )
        self.assertEqual(f.status, "fail")
        self.assertEqual(f.severity, "P0")
        self.assertEqual(f.risk_type, "覆盖性")
        self.assertIn("A1", f.evidence_refs)


if __name__ == "__main__":
    unittest.main()
