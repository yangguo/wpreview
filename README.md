---
title: workpaper NLP reviewer
emoji: 👁
colorFrom: indigo
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
---

# Workpaper NLP Reviewer

A comprehensive tool for reviewing workpapers and Excel files using AI/NLP techniques.

## Features

### 1. Web-based Workpaper Review (Streamlit App)
- Interactive web interface for reviewing audit procedures
- Supports Excel (.xlsx) and Word (.docx) files
- Multiple LLM model support (GPT-4, GPT-3.5, ERNIE, ChatGLM, etc.)
- Real-time review and feedback

### 2. Excel Image Review (Python Script) ⭐ NEW
- **Convert Excel sheets to images** for visual review
- **AI-powered analysis** using GPT-4 Vision
- **Comprehensive review** covering:
  - Spelling and grammatical errors
  - Logical inconsistencies
  - Data quality issues
  - Structural problems
- **Professional HTML reports** with images and detailed findings

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Excel Image Review (Standalone Script)

Convert Excel files to images and get AI-powered review:

```bash
python excel_image_review.py your_file.xlsx
```

See [EXCEL_REVIEW_README.md](EXCEL_REVIEW_README.md) for detailed documentation.

**Quick Start:**

1. Create a sample Excel file:
   ```bash
   python create_sample_excel.py
   ```

2. Review it:
   ```bash
   python excel_image_review.py sample_data.xlsx
   ```

3. Open `output/review_report.html` in your browser

### Web Application

Run the Streamlit app:

```bash
streamlit run app.py
```

## Configuration

### Environment Variables

Create a `.env` file with your API credentials:

```bash
AZURE_API_KEY=your_api_key_here
AZURE_BASE_URL=your_azure_endpoint_here
AZURE_DEPLOYMENT_NAME=your_deployment_name
AZURE_DEPLOYMENT_NAME_16K=your_16k_deployment_name
AZURE_DEPLOYMENT_NAME_GPT4=your_gpt4_deployment_name
AZURE_DEPLOYMENT_NAME_GPT4_32K=your_gpt4_32k_deployment_name
AZURE_DEPLOYMENT_NAME_GPT4_TURBO=your_gpt4_turbo_deployment_name
```

## Project Structure

```
├── app.py                      # Streamlit web application
├── excel_image_review.py       # Standalone Excel review script ⭐ NEW
├── create_sample_excel.py      # Create sample data for testing ⭐ NEW
├── test_excel_review.py        # Test suite for Excel review ⭐ NEW
├── checkwp.py                  # Workpaper review logic
├── gptfuc.py                   # GPT functions
├── utils.py                    # Utility functions
├── backend/                    # Backend services
│   ├── main.py
│   ├── corrector.py
│   └── txtencoder.py
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── EXCEL_REVIEW_README.md      # Detailed Excel review docs ⭐ NEW
```

## Key Features Explained

### Excel to Image Review

The new Excel review feature:
1. Reads Excel files using pandas and openpyxl
2. Converts each sheet to a high-quality PNG image
3. Uses GPT-4 Vision API to analyze the images
4. Generates an HTML report with:
   - Preview images of each sheet
   - Detailed findings and suggestions
   - Professional formatting

**What it detects:**
- Misspellings and typos
- Logical errors (e.g., negative quantities, date inconsistencies)
- Missing data
- Mathematical errors
- Structural issues

### Workpaper Review (Web App)

The Streamlit app provides:
- Upload Excel or Word files
- Select specific sheets/tables
- Choose review columns
- Get real-time AI feedback
- Download results as CSV

## Requirements

- Python 3.8+
- OpenAI API key or Azure OpenAI credentials
- See `requirements.txt` for Python packages

## Documentation

- [Excel Image Review Guide](EXCEL_REVIEW_README.md) - Comprehensive guide for the Excel review feature
- Hugging Face Space configuration (below)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See repository for license information.

---

# Hugging Face Space Configuration

`title`: _string_  
Display title for the Space

`emoji`: _string_  
Space emoji (emoji-only character allowed)

`colorFrom`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

`colorTo`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

`sdk`: _string_  
Can be either `gradio` or `streamlit`

`sdk_version` : _string_  
Only applicable for `streamlit` SDK.  
See [doc](https://hf.co/docs/hub/spaces) for more info on supported versions.

`app_file`: _string_  
Path to your main application file (which contains either `gradio` or `streamlit` Python code).  
Path is relative to the root of the repository.

`pinned`: _boolean_  
Whether the Space stays on top of your list.
