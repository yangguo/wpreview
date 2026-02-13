#!/usr/bin/env python3
"""
Test the Excel Image Reviewer with mock LLM responses.
This allows testing without making actual API calls.
"""

import os
import sys
from pathlib import Path

# Mock the LLM client before importing the main module
class MockCompletion:
    class Choice:
        class Message:
            def __init__(self):
                self.content = """## Review Results for Sheet

### 1. Spelling and Grammar
- **Issue Found**: "Enginering" should be "Engineering" in the Department column
- **Issue Found**: "Jonson" appears to be misspelled, should likely be "Johnson"

### 2. Logical Consistency
- All dates appear to be in logical order
- Salary values are within reasonable ranges
- Employee IDs are sequential

### 3. Data Quality Issues
- **Missing Data**: Employee ID 5 (Charlie Davis) has no salary value
- All other required fields are populated

### 4. Structural Issues
- Column headers are consistent and clear
- No unnecessary empty rows or columns
- Data types appear consistent within columns

### 5. Suggestions
1. Correct the spelling of "Enginering" to "Engineering"
2. Verify and correct "Jonson" to the proper spelling (likely "Johnson")
3. Fill in the missing salary for Charlie Davis (Employee ID 5)
4. Consider adding a validation rule to ensure all salary fields are populated
5. Consider standardizing date format if this will be shared across systems

Overall, the sheet structure is good with just a few data quality issues to address."""
        
        def __init__(self):
            self.message = self.Message()
    
    def __init__(self):
        self.choices = [self.Choice()]


class MockChatCompletions:
    def create(self, **kwargs):
        return MockCompletion()


class MockClient:
    def __init__(self, **kwargs):
        self.chat = type('obj', (object,), {'completions': MockChatCompletions()})()


# Monkey patch the Azure OpenAI client
import sys
from unittest.mock import MagicMock

sys.modules['openai'] = MagicMock()
sys.modules['openai'].AzureOpenAI = MockClient


# Now import and use the real module
from excel_image_review import ExcelImageReviewer


def test_excel_review():
    """Test the full Excel review workflow with mock LLM."""
    
    print("="*60)
    print("Testing Excel Image Reviewer (Mock Mode)")
    print("="*60)
    
    # Check if sample file exists
    excel_file = "sample_data.xlsx"
    if not os.path.exists(excel_file):
        print(f"Error: {excel_file} not found. Creating it...")
        import subprocess
        subprocess.run([sys.executable, "create_sample_excel.py"])
    
    # Create test output directory
    output_dir = "/tmp/excel_review_test"
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\nInput file: {excel_file}")
    print(f"Output directory: {output_dir}")
    
    # Create reviewer
    reviewer = ExcelImageReviewer(excel_file, output_dir, model_name="gpt-4-turbo")
    
    # Process all sheets
    print("\n" + "-"*60)
    reviewer.process_excel()
    
    # Generate report
    print("\n" + "-"*60)
    report_path = reviewer.generate_report()
    
    # Verify outputs
    print("\n" + "="*60)
    print("Verification:")
    print("="*60)
    
    # Check images
    for sheet_name, image_path in reviewer.sheet_images.items():
        if os.path.exists(image_path):
            size = os.path.getsize(image_path)
            print(f"✓ Image for '{sheet_name}': {image_path} ({size} bytes)")
        else:
            print(f"✗ Image for '{sheet_name}' not found!")
    
    # Check report
    if os.path.exists(report_path):
        size = os.path.getsize(report_path)
        print(f"✓ Report: {report_path} ({size} bytes)")
        
        # Show a preview of the report
        print("\nReport Preview (first 500 chars):")
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content[:500])
            print("...")
    else:
        print(f"✗ Report not found!")
    
    # Check reviews
    print(f"\n✓ Total reviews generated: {len(reviewer.sheet_reviews)}")
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60)
    print(f"\nYou can view the report at: {report_path}")
    
    return reviewer, report_path


if __name__ == "__main__":
    reviewer, report_path = test_excel_review()
