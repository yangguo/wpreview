#!/usr/bin/env python3
"""Example usage of the structured Excel workpaper reviewer."""

import os
import subprocess
import sys

from excel_image_review import ExcelImageReviewer


def example_basic_usage():
    print("Example 1: Structured Review")
    print("-" * 60)

    reviewer = ExcelImageReviewer(
        excel_path="sample_data.xlsx",
        output_dir="output",
        model_name="gpt-4o",
    )

    reviewer.process_excel(limit=1)
    report_path = reviewer.generate_report()

    print(f"Report generated: {report_path}")
    return reviewer


def example_access_results():
    print("\nExample 2: Accessing Structured Results")
    print("-" * 60)

    reviewer = ExcelImageReviewer(
        excel_path="sample_data.xlsx",
        output_dir="output",
    )

    reviewer.process_excel(limit=1)

    print(f"\nSheets processed: {len(reviewer.sheet_reviews)}")

    for sheet_name, review_text in reviewer.sheet_reviews.items():
        print(f"\n{'=' * 60}")
        print(f"Sheet: {sheet_name}")
        print(f"{'=' * 60}")

        structure = reviewer.sheet_structures.get(sheet_name, {})
        records = reviewer.sheet_schema_records.get(sheet_name, [])
        print(f"Non-empty cells: {structure.get('non_empty_cell_count', 0)}")
        print(f"Merged ranges: {len(structure.get('merged_ranges', []))}")
        print(f"Schema records: {len(records)}")
        print("Review (first 300 chars):")
        print((review_text or "")[:300] + "...")

    return reviewer


def main():
    print("=" * 60)
    print("Excel Structured Reviewer - Usage Examples")
    print("=" * 60)

    if not os.path.exists("sample_data.xlsx"):
        print("\nSample file not found. Creating it...")
        subprocess.run([sys.executable, "create_sample_excel.py"], check=True)

    try:
        example_basic_usage()
        # example_access_results()

        print("\n" + "=" * 60)
        print("All examples completed")
        print("=" * 60)
    except Exception as exc:
        print(f"\nError: {exc}")
        raise


if __name__ == "__main__":
    main()
