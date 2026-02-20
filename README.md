# Workpaper NLP Reviewer

Reviews Excel workpapers by converting each sheet to an image and analyzing them with an LLM vision model.

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
```
  excel_file        Path to the Excel file
  -o, --output DIR  Output directory (default: output)
  -m, --model MODEL Model to use (default: gpt-4o)
```

Each sheet is converted to an image and reviewed for spelling errors, logical inconsistencies, data quality issues, and structural problems. Results are saved as an HTML report in the output directory.

To test with sample data:
```bash
python create_sample_excel.py
python excel_image_review.py sample_data.xlsx
# open output/review_report.html
```

Programmatic usage:
```python
from excel_image_review import ExcelImageReviewer

reviewer = ExcelImageReviewer("data.xlsx", output_dir="results")
reviewer.process_excel()
report_path = reviewer.generate_report()
```

## Project Structure

```
├── excel_image_review.py   # Main script
├── create_sample_excel.py  # Sample data generator
├── test_excel_review.py    # Tests
└── requirements.txt
```

## License

See repository for license information.

