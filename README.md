# Automated TTB Compliance Processing - Deployment Guide

## Overview
A Streamlit web application that uses OCR to verify beer label compliance with TTB (Alcohol and Tobacco Tax and Trade Bureau) regulations.

## Prerequisites
- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and available on your system PATH

### Installing Tesseract

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download the installer from [UB Mannheim's Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your PATH.

## Python Dependencies

Create a `requirements.txt` with the following:

```txt
streamlit
pandas
numpy
easyocr
pytesseract
opencv-python
rapidfuzz
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

## File Structure

Ensure the following files are present in the project root:

```
├── app.py            # Streamlit web application
├── cleaning.py       # Utility to reset processing results
├── data.json         # Application records (label metadata)
├── parsing.py        # Label element detection and verification
├── processing.py     # Main processing and evaluation logic
├── scanning.py       # OCR scanning (Tesseract + EasyOCR)
├── utils.py          # Shared helper functions
└── test_labels/      # Directory containing label images (.png)
    ├── CJ177443.png
    ├── CL507720.png
    └── ...
```

### Important Notes
- Label images must be `.png` files named by their application number (e.g., `CJ177443.png`) [1].
- `data.json` must contain entries whose `application_num` values match the image filenames [3].
- Only entries with a corresponding image in `test_labels/` will be processable [1].

## Running the Application

```bash
streamlit run app.py
```

## Resetting Results

To clear all processing results from `data.json`, either:
- Use the **Reset Results** button in the app, or
- Run the cleaning script directly:

```bash
python cleaning.py
```

Both methods remove the `processing_result` field from every entry in `data.json` [2].

## Usage
1. **Preview Labels** — Search and preview label images in the top section [1].
2. **Process Labels** — Select labels from the table and click "Process Selected" to run OCR validation [1].
3. **Download Results** — After processing, download a CSV summary of pass/fail results [1].
4. **Reset** — Clear all results to reprocess labels.