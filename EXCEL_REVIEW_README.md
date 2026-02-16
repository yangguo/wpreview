# Excel Image Reviewer

A Python script that converts Excel sheets to images and uses LLM (Large Language Model) with vision capabilities to review them for errors, inconsistencies, and quality issues.

## Features

- **Excel to Image Conversion**: Converts each sheet in an Excel file to a high-quality image
- **AI-Powered Review**: Uses GPT-4 Vision (or similar models) to analyze each sheet image
- **Comprehensive Analysis**: Identifies:
  - Spelling and grammatical errors
  - Logical inconsistencies
  - Data quality issues
  - Structural problems
- **Professional Report**: Generates an HTML report with images and detailed findings

## Requirements

### Python Packages

```bash
pip install -r requirements.txt
```

Required packages:
- `openpyxl` - For reading Excel files
- `pandas` - For data manipulation
- `pillow` - For image processing
- `openai` - For LLM integration
- `python-dotenv` - For environment variables

### Environment Variables

Create a `.env` file in the project root with your Azure OpenAI credentials:

```bash
AZURE_API_KEY=your_api_key_here
AZURE_BASE_URL=your_azure_endpoint_here
AZURE_DEPLOYMENT_NAME_GPT4_TURBO=your_deployment_name
```

## Usage

### Basic Usage

```bash
python excel_image_review.py path/to/your/file.xlsx
```

### With Options

```bash
python excel_image_review.py path/to/your/file.xlsx -o custom_output_dir -m gpt-4-turbo
```

### Command Line Arguments

- `excel_file` (required): Path to the Excel file to review
- `-o, --output` (optional): Output directory for images and report (default: "output")
- `-m, --model` (optional): LLM model to use (default: "gpt-4-turbo")

## Example

### 1. Create a Sample Excel File

```bash
python create_sample_excel.py
```

This creates `sample_data.xlsx` with intentional errors for testing.

### 2. Review the Excel File

```bash
python excel_image_review.py sample_data.xlsx
```

### 3. View the Report

Open `output/review_report.html` in your web browser to see:
- Images of each sheet
- Detailed AI analysis of each sheet
- Specific findings and suggestions

## Output Structure

```
output/
├── Employee Data_screenshot.png
├── Sales Data_screenshot.png
├── Budget Summary_screenshot.png
└── review_report.html
```

## What the Review Covers

### 1. Spelling and Grammar
- Misspellings in headers, labels, or data
- Typos and grammatical errors
- Inconsistent terminology

### 2. Logical Consistency
- Dates that don't make sense (e.g., end date before start date)
- Numbers out of expected ranges
- Inconsistent categorizations
- Mathematical errors in calculations

### 3. Data Quality Issues
- Missing or incomplete data
- Duplicate entries
- Inconsistent formatting
- Values that don't match expected patterns

### 4. Structural Issues
- Inconsistent column headers
- Problematic merged cells
- Unnecessary empty rows or columns
- Layout problems

### 5. Suggestions
- Specific, actionable recommendations
- Best practices for data organization
- Ways to improve clarity and usability

## How It Works

1. **Load Excel File**: Reads the Excel file using pandas and openpyxl
2. **Convert to Images**: Each sheet is converted to a PNG image showing the data in a table format
3. **AI Analysis**: Each image is sent to GPT-4 Vision API for analysis
4. **Generate Report**: Creates an HTML report combining images and review text

## Limitations

- Very large Excel files may take longer to process
- Complex formatting (colors, conditional formatting) is simplified in images
- Charts and embedded objects are not included in the basic conversion
- API costs apply for LLM usage (Azure OpenAI or OpenAI)

## Troubleshooting

### Error: "AZURE_API_KEY not found"
- Make sure you have created a `.env` file with your Azure OpenAI credentials

### Error: "File not found"
- Check that the Excel file path is correct
- Use absolute paths or ensure you're running from the correct directory

### Poor Image Quality
- For very wide spreadsheets, images may be compressed
- Consider splitting large sheets into smaller ones for better review

### LLM Timeout
- Large images may take longer to process
- Try reducing the sheet size or splitting into multiple files

## Advanced Usage

### Programmatic Usage

You can also use the `ExcelImageReviewer` class in your own Python scripts:

```python
from excel_image_review import ExcelImageReviewer

# Create reviewer
reviewer = ExcelImageReviewer("data.xlsx", output_dir="results")

# Process all sheets
reviewer.process_excel()

# Generate report
report_path = reviewer.generate_report()
print(f"Report generated: {report_path}")
```

### Customizing the Review Prompt

Edit the `review_image()` method in `excel_image_review.py` to customize what aspects the LLM should focus on.

## Tips for Best Results

1. **Clean Data First**: Remove obviously empty rows/columns before review
2. **One Topic Per Sheet**: Keep related data together in single sheets
3. **Clear Headers**: Use descriptive column headers
4. **Review in Batches**: For large workbooks, review a few sheets at a time
5. **Verify Findings**: Always verify AI suggestions - they may occasionally be incorrect

## License

This tool is part of the wpreview project. See the main repository for license information.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
