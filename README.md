# Workpaper NLP Reviewer

Reviews Excel audit workpapers by directly reading workbook structure with `openpyxl` (no image recognition), then using an LLM for methodology and consistency review.

## Architecture

1. Document preprocessing (structured extraction)
- Read cell coordinates, row/column positions, and merged ranges via `openpyxl`
- Detect contiguous table regions
- Map fields into a standard Schema:
  - `control_id`
  - `control_description`
  - `audit_objective`
  - `audit_procedure`
  - `sample_selection_method`
  - `sample_size`
  - `test_steps`
  - `test_result`
  - `exception_flag`
  - `conclusion`

2. LLM review layer
- Coverage: whether test execution covers audit procedures
- Methodology issues: evidence sufficiency, sampling basis, sample size, exception closure, conclusion-evidence alignment
- Logical consistency: internal contradictions
- Cross-field consistency: control/procedure/test/result/conclusion/exception/date-period alignment

3. Report generation
- Output issue list, location, source quote, judgement basis, and remediation advice
- Save structured JSON per sheet and final DOCX report

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
```

## Usage

```bash
python excel_image_review.py path/to/file.xlsx
python excel_image_review.py path/to/file.xlsx -o output_dir -m gpt-4o
```

Options:

```text
  excel_file        Path to the Excel file
  -o, --output DIR  Output directory (default: output)
  -m, --model MODEL Model to use (default: gpt-4o)
```

Outputs:
- `output/<sheet_name>_structured.json` (structured extraction payload)
- `output/review_report.docx` (review report)

To test with sample data:

```bash
python create_sample_excel.py
python excel_image_review.py sample_data.xlsx
```

Programmatic usage:

```python
from excel_image_review import ExcelImageReviewer

reviewer = ExcelImageReviewer("data.xlsx", output_dir="results")
reviewer.process_excel()
report_path = reviewer.generate_report()
```

## Project Structure

```text
├── excel_image_review.py                 # Main script (structured reviewer)
├── create_sample_excel.py                # Sample data generator
├── test_excel_review.py                  # End-to-end mock test
├── test_excel_image_review_libreoffice.py # Structure/schema unit tests
└── requirements.txt
```

## License

See repository for license information.
