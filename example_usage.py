#!/usr/bin/env python3
"""
Example usage of the Excel Image Reviewer.

This script demonstrates how to use the ExcelImageReviewer class
programmatically in your own Python applications.
"""

from excel_image_review import ExcelImageReviewer
import sys


def example_basic_usage():
    """Basic usage example."""
    print("Example 1: Basic Usage")
    print("-" * 60)
    
    # Create a reviewer instance
    reviewer = ExcelImageReviewer(
        excel_path="sample_data.xlsx",
        output_dir="output",
        model_name="gpt-4-turbo"
    )
    
    # Process all sheets
    reviewer.process_excel()
    
    # Generate report
    report_path = reviewer.generate_report()
    
    print(f"Report generated: {report_path}")
    return reviewer


def example_access_results():
    """Example showing how to access review results."""
    print("\nExample 2: Accessing Results Programmatically")
    print("-" * 60)
    
    reviewer = ExcelImageReviewer(
        excel_path="sample_data.xlsx",
        output_dir="output"
    )
    
    reviewer.process_excel()
    
    # Access the results
    print(f"\nSheets processed: {len(reviewer.sheet_reviews)}")
    
    for sheet_name, review_text in reviewer.sheet_reviews.items():
        print(f"\n{'='*60}")
        print(f"Sheet: {sheet_name}")
        print(f"{'='*60}")
        print(f"Image: {reviewer.sheet_images.get(sheet_name)}")
        print(f"\nReview (first 200 chars):")
        print(review_text[:200] + "...")
    
    return reviewer


def example_custom_processing():
    """Example with custom processing logic."""
    print("\nExample 3: Custom Processing")
    print("-" * 60)
    
    reviewer = ExcelImageReviewer(
        excel_path="sample_data.xlsx",
        output_dir="custom_output"
    )
    
    # Process only specific sheets
    import pandas as pd
    xls = pd.ExcelFile("sample_data.xlsx")
    
    for sheet_name in xls.sheet_names:
        if "Employee" in sheet_name:  # Only process sheets with "Employee" in name
            print(f"\nProcessing: {sheet_name}")
            image = reviewer.excel_to_image(sheet_name)
            
            if image:
                # Save image
                from pathlib import Path
                image_path = reviewer.output_dir / f"{sheet_name}_custom.png"
                image.save(image_path)
                reviewer.sheet_images[sheet_name] = image_path
                
                # Review image
                review = reviewer.review_image(image, sheet_name)
                reviewer.sheet_reviews[sheet_name] = review
                
                print(f"✓ Processed {sheet_name}")
    
    # Generate report with custom sheets
    report_path = reviewer.generate_report()
    print(f"\nCustom report generated: {report_path}")
    
    return reviewer


def main():
    """Run all examples."""
    print("="*60)
    print("Excel Image Reviewer - Usage Examples")
    print("="*60)
    
    # Check if sample file exists
    import os
    if not os.path.exists("sample_data.xlsx"):
        print("\nSample file not found. Creating it...")
        import subprocess
        subprocess.run([sys.executable, "create_sample_excel.py"])
    
    print("\n")
    
    # Run examples
    try:
        # Example 1: Basic usage
        example_basic_usage()
        
        # Example 2: Access results
        # Uncomment to run:
        # example_access_results()
        
        # Example 3: Custom processing
        # Uncomment to run:
        # example_custom_processing()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
