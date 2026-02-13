#!/usr/bin/env python3
"""
Excel Image Reviewer
Convert Excel sheets to images and use LLM to review for misspellings, 
logical problems, inconsistencies, etc.
"""

import argparse
import base64
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd
from dotenv import load_dotenv
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from PIL import Image, ImageDraw, ImageFont

# Load environment variables
load_dotenv()


class ExcelImageReviewer:
    """Convert Excel sheets to images and review them using LLM."""

    def __init__(self, excel_path, output_dir="output", model_name="gpt-4-turbo"):
        """
        Initialize the Excel Image Reviewer.

        Args:
            excel_path: Path to the Excel file
            output_dir: Directory to save output files
            model_name: LLM model to use for review
        """
        self.excel_path = excel_path
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.output_dir.mkdir(exist_ok=True)
        
        # Store for images and reviews
        self.sheet_images = {}
        self.sheet_reviews = {}
        
        # Initialize LLM
        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM client."""
        try:
            from openai import AzureOpenAI
            
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version="2024-02-15-preview",
                azure_endpoint=os.getenv("AZURE_BASE_URL"),
            )
            self.deployment = os.getenv("AZURE_DEPLOYMENT_NAME_GPT4_TURBO", "gpt-4-turbo")
        except ImportError:
            print("Error: openai package not installed. Please install it.")
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            sys.exit(1)

    def excel_to_image(self, sheet_name):
        """
        Convert an Excel sheet to an image.

        Args:
            sheet_name: Name of the sheet to convert

        Returns:
            PIL Image object
        """
        # Read the Excel sheet
        df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
        
        # Handle empty or invalid data
        if df.empty:
            print(f"Warning: Sheet '{sheet_name}' is empty")
            return None
        
        # Fill NaN values
        df = df.fillna("")
        
        # Convert to string representation
        df_str = df.astype(str)
        
        # Create image from dataframe
        image = self._dataframe_to_image(df_str, sheet_name)
        
        return image

    def _dataframe_to_image(self, df, title):
        """
        Convert a pandas DataFrame to an image.

        Args:
            df: pandas DataFrame
            title: Title for the image

        Returns:
            PIL Image object
        """
        # Calculate image dimensions
        num_rows, num_cols = df.shape
        
        # Settings for cell dimensions
        cell_width = 120
        cell_height = 30
        header_height = 40
        title_height = 50
        padding = 10
        
        # Calculate total dimensions
        img_width = max(800, num_cols * cell_width + padding * 2)
        img_height = title_height + header_height + (num_rows * cell_height) + padding * 2
        
        # Create image
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a better font
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            cell_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except (IOError, OSError):
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            cell_font = ImageFont.load_default()
        
        # Draw title
        draw.text((padding, padding), title, fill='black', font=title_font)
        
        # Starting Y position for table
        y_offset = title_height + padding
        
        # Draw header row (column names)
        x_offset = padding
        for col in df.columns:
            # Draw header cell background
            draw.rectangle(
                [(x_offset, y_offset), (x_offset + cell_width, y_offset + header_height)],
                fill='lightblue',
                outline='black'
            )
            # Draw header text
            text = str(col)[:15]  # Truncate long headers
            draw.text((x_offset + 5, y_offset + 10), text, fill='black', font=header_font)
            x_offset += cell_width
        
        y_offset += header_height
        
        # Draw data rows
        for idx, row in df.iterrows():
            x_offset = padding
            for col in df.columns:
                # Draw cell background
                draw.rectangle(
                    [(x_offset, y_offset), (x_offset + cell_width, y_offset + cell_height)],
                    fill='white',
                    outline='gray'
                )
                # Draw cell text
                text = str(row[col])[:20]  # Truncate long text
                draw.text((x_offset + 5, y_offset + 8), text, fill='black', font=cell_font)
                x_offset += cell_width
            y_offset += cell_height
        
        return img

    def review_image(self, image, sheet_name):
        """
        Review an image using LLM vision capabilities.

        Args:
            image: PIL Image object
            sheet_name: Name of the sheet being reviewed

        Returns:
            Review text from LLM
        """
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare the prompt
        prompt = f"""You are a professional data quality reviewer. Please carefully review this spreadsheet image (sheet: {sheet_name}) and provide a detailed analysis covering:

1. **Spelling and Grammar**: Identify any misspellings, typos, or grammatical errors in headers, labels, or data entries.

2. **Logical Consistency**: Check for logical inconsistencies, such as:
   - Dates that don't make sense (e.g., end date before start date)
   - Numbers that seem out of range or illogical
   - Inconsistent categorization or classifications

3. **Data Quality Issues**: Look for:
   - Missing or incomplete data
   - Duplicate entries
   - Inconsistent formatting (e.g., date formats, number formats)
   - Values that don't match expected patterns

4. **Structural Issues**: Check for:
   - Inconsistent column headers
   - Merged cells that may cause confusion
   - Empty rows or columns that shouldn't be there

5. **Suggestions**: Provide specific, actionable suggestions for improvement.

Please provide your review in a structured format with clear sections and specific examples where issues are found."""

        try:
            # Call Azure OpenAI API with vision
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional data quality and spreadsheet reviewer with expertise in identifying errors, inconsistencies, and suggesting improvements."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            review_text = response.choices[0].message.content
            return review_text
            
        except Exception as e:
            error_msg = f"Error reviewing image: {str(e)}"
            print(error_msg)
            return error_msg

    def process_excel(self):
        """Process all sheets in the Excel file."""
        print(f"Processing Excel file: {self.excel_path}")
        
        # Get all sheet names
        try:
            xls = pd.ExcelFile(self.excel_path)
            sheet_names = xls.sheet_names
            print(f"Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return
        
        # Process each sheet
        for sheet_name in sheet_names:
            print(f"\nProcessing sheet: {sheet_name}")
            
            # Convert sheet to image
            try:
                image = self.excel_to_image(sheet_name)
                if image is None:
                    continue
                
                # Save image
                image_path = self.output_dir / f"{sheet_name}_screenshot.png"
                image.save(image_path)
                print(f"  - Image saved: {image_path}")
                
                self.sheet_images[sheet_name] = image_path
                
            except Exception as e:
                print(f"  - Error converting sheet to image: {e}")
                continue
            
            # Review image
            try:
                print(f"  - Reviewing with LLM...")
                review = self.review_image(image, sheet_name)
                self.sheet_reviews[sheet_name] = review
                print(f"  - Review completed")
                
            except Exception as e:
                print(f"  - Error reviewing image: {e}")
                self.sheet_reviews[sheet_name] = f"Error: {str(e)}"

    def generate_report(self):
        """Generate a comprehensive HTML report."""
        print("\nGenerating report...")
        
        # Create HTML report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Excel Review Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
        }}
        .metadata {{
            color: #ecf0f1;
            font-size: 14px;
            margin-top: 10px;
        }}
        .sheet-section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .sheet-title {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}
        .image-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
        .review-content {{
            line-height: 1.6;
            color: #333;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .summary {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Excel Review Report</h1>
        <div class="metadata">
            <p><strong>File:</strong> {os.path.basename(self.excel_path)}</p>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Sheets Reviewed:</strong> {len(self.sheet_reviews)}</p>
        </div>
    </div>
    
    <div class="summary">
        <h2>📋 Summary</h2>
        <p>This report contains automated reviews of {len(self.sheet_reviews)} sheet(s) from the Excel file. 
        Each sheet has been converted to an image and analyzed by an AI model for potential issues including 
        spelling errors, logical inconsistencies, data quality problems, and structural issues.</p>
    </div>
"""
        
        # Add each sheet's review
        for sheet_name in self.sheet_reviews.keys():
            image_path = self.sheet_images.get(sheet_name)
            review = self.sheet_reviews.get(sheet_name, "No review available")
            
            # Convert image path to relative path for HTML
            if image_path:
                rel_image_path = os.path.basename(image_path)
            else:
                rel_image_path = ""
            
            html_content += f"""
    <div class="sheet-section">
        <h2 class="sheet-title">📄 Sheet: {sheet_name}</h2>
        
        <div class="image-container">
            <h3>Sheet Preview</h3>
            <img src="{rel_image_path}" alt="{sheet_name} preview">
        </div>
        
        <h3>🔍 Review Results</h3>
        <div class="review-content">{review}</div>
    </div>
"""
        
        html_content += """
    <div class="footer">
        <p>Generated by Excel Image Reviewer</p>
        <p>Powered by AI-based analysis</p>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        report_path = self.output_dir / "review_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report saved: {report_path}")
        return report_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert Excel sheets to images and review them using LLM"
    )
    parser.add_argument(
        "excel_file",
        help="Path to the Excel file to review"
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory for images and report (default: output)"
    )
    parser.add_argument(
        "-m", "--model",
        default="gpt-4-turbo",
        help="LLM model to use (default: gpt-4-turbo)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.excel_file):
        print(f"Error: File not found: {args.excel_file}")
        sys.exit(1)
    
    # Check if file is Excel
    if not args.excel_file.endswith(('.xlsx', '.xls')):
        print("Error: File must be an Excel file (.xlsx or .xls)")
        sys.exit(1)
    
    # Create reviewer and process
    reviewer = ExcelImageReviewer(args.excel_file, args.output, args.model)
    
    try:
        # Process all sheets
        reviewer.process_excel()
        
        # Generate report
        report_path = reviewer.generate_report()
        
        print("\n" + "="*60)
        print("✅ Review completed successfully!")
        print(f"📁 Output directory: {reviewer.output_dir}")
        print(f"📄 Report: {report_path}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
