#!/usr/bin/env python3
"""Excel Image Reviewer - Convert Excel sheets to images and review with LLM vision."""

import argparse
import base64
import io
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import openpyxl
from openpyxl.utils import get_column_letter

load_dotenv()


class ExcelImageReviewer:
    """Convert Excel sheets to images and review them using LLM vision."""

    def __init__(self, excel_path, output_dir="output", model_name=None, base_url=None):
        self.excel_path = excel_path
        self.output_dir = Path(output_dir)
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.output_dir.mkdir(exist_ok=True)
        self.sheet_images = {}
        self.sheet_reviews = {}
        client_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        resolved_url = base_url or os.getenv("OPENAI_BASE_URL")
        if resolved_url:
            client_kwargs["base_url"] = resolved_url
        self.client = OpenAI(**client_kwargs)

    def excel_to_image(self, sheet_name):
        """Convert an Excel sheet to a PIL Image covering only the populated content area."""
        wb = openpyxl.load_workbook(self.excel_path, data_only=True)
        ws = wb[sheet_name]

        if ws.max_row is None or ws.max_column is None:
            print(f"Warning: Sheet '{sheet_name}' is empty")
            return None

        min_r, max_r = ws.min_row, ws.max_row
        min_c, max_c = ws.min_column, ws.max_column

        # Trim trailing blank rows
        while max_r >= min_r:
            if any(ws.cell(max_r, c).value not in (None, "") for c in range(min_c, max_c + 1)):
                break
            max_r -= 1

        # Trim trailing blank columns
        while max_c >= min_c:
            if any(ws.cell(r, max_c).value not in (None, "") for r in range(min_r, max_r + 1)):
                break
            max_c -= 1

        if max_r < min_r or max_c < min_c:
            print(f"Warning: Sheet '{sheet_name}' is empty")
            return None

        rows = [
            [str(ws.cell(r, c).value) if ws.cell(r, c).value is not None else ""
             for c in range(min_c, max_c + 1)]
            for r in range(min_r, max_r + 1)
        ]

        col_widths = []
        for c in range(min_c, max_c + 1):
            dim = ws.column_dimensions.get(get_column_letter(c))
            w = dim.width if (dim and dim.width) else 10
            col_widths.append(max(80, int(w * 7)))

        row_heights = []
        for r in range(min_r, max_r + 1):
            dim = ws.row_dimensions.get(r)
            h = dim.height if (dim and dim.height) else 15
            row_heights.append(max(22, int(h * 1.33)))

        return self._render_table_image(rows, col_widths, row_heights, sheet_name)

    def _render_table_image(self, rows, col_widths, row_heights, title):
        """Render a 2D list of cell strings as a PIL Image table."""
        PADDING, TITLE_H = 10, 35
        total_w = PADDING * 2 + sum(col_widths)
        total_h = PADDING + TITLE_H + sum(row_heights) + PADDING

        img = Image.new("RGB", (total_w, total_h), "white")
        draw = ImageDraw.Draw(img)

        def load_font(candidates, size):
            for path in candidates:
                if os.path.exists(path):
                    try:
                        return ImageFont.truetype(path, size)
                    except Exception:
                        pass
            return ImageFont.load_default()

        bold_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
        reg_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        title_font = load_font(bold_candidates, 13)
        bold_font = load_font(bold_candidates, 11)
        cell_font = load_font(reg_candidates, 10)

        draw.text((PADDING, PADDING), title, fill="#2c3e50", font=title_font)
        y = PADDING + TITLE_H

        for row_idx, (row, rh) in enumerate(zip(rows, row_heights)):
            x = PADDING
            is_header = row_idx == 0
            for cell_val, cw in zip(row, col_widths):
                fill = "#cfe2f3" if is_header else ("white" if row_idx % 2 == 0 else "#f5f5f5")
                draw.rectangle([(x, y), (x + cw - 1, y + rh - 1)], fill=fill, outline="#bbbbbb")
                font = bold_font if is_header else cell_font
                max_chars = max(4, (cw - 8) // 6)
                text = cell_val[:max_chars] + ("\u2026" if len(cell_val) > max_chars else "")
                draw.text((x + 4, y + max(2, (rh - 12) // 2)), text, fill="#1a1a1a", font=font)
                x += cw
            y += rh

        return img

    def review_image(self, image, sheet_name):
        """Review a sheet image using LLM vision and return the review text."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        prompt = (
            f"You are a professional data quality reviewer. Carefully review this "
            f"spreadsheet image (sheet: {sheet_name}) and provide a detailed analysis covering:\n\n"
            "1. **Spelling and Grammar**: Misspellings, typos, or grammatical errors in headers or data.\n"
            "2. **Logical Consistency**: Illogical dates, out-of-range numbers, inconsistent categorizations.\n"
            "3. **Data Quality**: Missing data, duplicates, inconsistent formatting.\n"
            "4. **Structural Issues**: Inconsistent headers, problematic merged cells, unnecessary empty rows/columns.\n"
            "5. **Suggestions**: Specific, actionable recommendations.\n\n"
            "Provide your review in a structured format with clear sections and specific examples."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional data quality and spreadsheet reviewer.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        ],
                    },
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error reviewing image: {e}")
            return f"Error reviewing image: {e}"

    def _excel_to_images_libreoffice(self, sheet_names):
        """Convert all sheets via LibreOffice headless → PDF → PIL Images.
        Returns {sheet_name: PIL.Image} mapped by sheet order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf",
                 "--outdir", tmpdir, str(self.excel_path)],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode().strip() or "LibreOffice conversion failed")
            pdf_path = Path(tmpdir) / (Path(self.excel_path).stem + ".pdf")
            if not pdf_path.exists():
                raise RuntimeError(f"Expected PDF not found: {pdf_path}")
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path, dpi=150)
            return {name: page for name, page in zip(sheet_names, pages)}

    def process_excel(self):
        """Convert all sheets to images and review each one."""
        print(f"Processing: {self.excel_path}")
        try:
            sheet_names = pd.ExcelFile(self.excel_path).sheet_names
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return
        print(f"Found {len(sheet_names)} sheet(s): {', '.join(sheet_names)}")

        # Try LibreOffice for high-fidelity rendering; fall back to built-in PIL renderer
        libreoffice_images = {}
        if shutil.which("soffice"):
            try:
                print("Converting with LibreOffice...")
                libreoffice_images = self._excel_to_images_libreoffice(sheet_names)
                print("LibreOffice conversion successful.")
            except Exception as e:
                print(f"LibreOffice failed ({e}), falling back to built-in renderer")
        else:
            print("LibreOffice not found, using built-in PIL renderer")

        for sheet_name in sheet_names:
            print(f"\n[{sheet_name}] Converting to image...")
            try:
                image = libreoffice_images.get(sheet_name) or self.excel_to_image(sheet_name)
                if image is None:
                    continue
                image_path = self.output_dir / f"{sheet_name}_screenshot.png"
                image.save(image_path)
                self.sheet_images[sheet_name] = image_path
                print(f"[{sheet_name}] Reviewing with LLM...")
                self.sheet_reviews[sheet_name] = self.review_image(image, sheet_name)
                print(f"[{sheet_name}] Done.")
            except Exception as e:
                print(f"[{sheet_name}] Error: {e}")
                self.sheet_reviews[sheet_name] = f"Error: {e}"

    def generate_report(self):
        """Generate an HTML report with sheet images and review results."""
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
        
        report_path = self.output_dir / "review_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Report saved: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Excel sheets to images and review with LLM vision"
    )
    parser.add_argument("excel_file", help="Path to the Excel file")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("-m", "--model", default=None, help="Model to use (default: OPENAI_MODEL env or gpt-4o)")
    parser.add_argument("-u", "--url", default=None, help="Base URL for OpenAI-compatible API endpoint (default: OPENAI_BASE_URL env)")
    args = parser.parse_args()

    if not os.path.exists(args.excel_file):
        print(f"Error: File not found: {args.excel_file}")
        sys.exit(1)
    if not args.excel_file.endswith((".xlsx", ".xls")):
        print("Error: File must be an Excel file (.xlsx or .xls)")
        sys.exit(1)

    reviewer = ExcelImageReviewer(args.excel_file, args.output, args.model, args.url)
    reviewer.process_excel()
    report_path = reviewer.generate_report()
    print(f"\nDone! Report: {report_path}")


if __name__ == "__main__":
    main()
