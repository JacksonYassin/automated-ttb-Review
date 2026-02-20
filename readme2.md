# Deep Learning Fusion OCR Model for Automated TTB Label Compliance Verificiation

This automated TTB-compliance checker for malt beverages employs two, distinct optical character recognition models to extract text from labels: [Tesseract](https://github.com/tesseract-ocr/tesseract) and [EasyOCR](https://github.com/JaidedAI/EasyOCR). The extracted text is then cleaned and processed and the results from the two different scans are fused to improve the quality of the OCR. The scanned text is then checked against the submitted application to verify accuracy and correctness. The architecture is designed to minimize false positives. A streamlit web application provides a clean user interface to engage with labels for processing.


## Design overview

### Image Processing Pipeline
1. An image of a label is passed into `process_label`.
2. The image is passed to `tesseract` and `easyOCR` for scanning. Text, along with locations of the text and the model's confidence in its output, are loaded into a `pandas` dataframe. 
3. The two different lists are compared to find similar words. To find the highest-quality representation of the text, for each pair, the word with the higher model confidence score is passed into the output `fused_set`. 
4. All text is now scanned from the image. It is then compared with the application information to verify accuracy using `find_elements`.  
5. For the elements not present in the application (alcohol content, net contents, and government warning), three seperate checks are employed to verify accuracy and presence on the label. 
6. All relevant information on the label is now processed and verified for accuracy. If all information is present and accurately represented, the final output list will contain the information and its location on the label. If the label failed, one or more of the eight features will have "None" in its output place.  


### OCR methods
Both Tesseract and EasyOCR employ Convolutional Recurrent Neural Networks (CRNNs) to detect text in images. There are two key benefits to using a deep learning based architecture:
1. **Efficiency**: Processing time for an image is ~2 seconds compared to ~20-40+ seconds for LLM-based methods.

2. **Low false positive rates**: Hallucinations are not a feature of deep learning models. While they might misclassify text, comparison to ground-truth labels and the fusion technique makes false positives almost impossible, an important feature given the danger of false positives in this application. 

Both Tesseract and EasyOCR provide relatively accurate outputs. However, as they have different approaches to detecting text, they usually have some disagreements and given the nature of the models, often will make errors. However, the different approaches mean that errors are usually made in different places by the two models. Even if one model got it wrong, the other likely got it right. Using these two scanning methods together provides a highly accurate scan of the text in the image.


### Streamlit Web App
Functionality is hosted through a Streamlit web app. This app allows users to easily preview the labels, as well as run in individual and batch processing commands. Streamlit's implementation is something like feature-rich, python implementation of html, meaning elements are created top to bottom in the code. 

### Code
#### `scanning.py`
The functions in this file are the core driver functions to extract text from an image. `ocr_scan` provides the tesseract and easyOCR calls to get the initial scanned text while `make_fusion_list` and `fuse_lists` provides voting to select the best candidates for each found word. Together, these functions take in an image and output a list of words with their locations and model confidence in the found words.

#### `parsing.py`
The functions in this file go through the found words to verify that all relevant information is present. They differ slighly in implementation. `find_elements` is able to verify found words though comparison with the ground-truth application data. `verify_locations` ensures that alcohol content and net contents correctly displays information by finding spatially proximate data (i.e. paragraphs/text blocks) using DBSCAN and checking formatting within these paragraphs. `verify_government_warning` enforces strict punctuation, capitalization, and wording requirements on the found text to ensure compliance with the government label. *note: this check is unable to verify bold font.*

#### `processing.py`
The functions in this file orchestrate the scanning, parsing, and validation of found data. Data search and validation is performed by `process_label` while the results are interpreted by `evaluate_label_results`.


## Local setup

### 1. Prerequisites

### System requirements

* **Python 3.10–3.12** (I recommended Python 3.10 to avoid potential errors from Streamlit due to lack of support for later Python releases)


### System packages (required for OCR)

Install **Tesseract OCR** and **OpenCV dependencies** at the system level.

#### macOS (Homebrew)

```bash
brew install tesseract
brew install opencv
```

Verify Tesseract is installed:

```bash
tesseract --version
```

## 2. Create a Python Virtual Environment

Run in terminal from the root of the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Python Dependencies (One Command)

All required Python packages can be installed with the following command:

```bash
pip install \
  streamlit \
  pandas \
  easyocr \
  pytesseract \
  opencv-python \
  rapidfuzz \
  matplotlib
```

They can also be installed individually:
```bash
pip install streamlit
pip install pandas
...
```

### Notes on dependencies

* **easyocr** will automatically download model weights on first run
* **pytesseract** is just a Python wrapper — the Tesseract binary must already be installed
* **opencv-python** is used for image loading and preprocessing



## 4. Project Structure Overview

```text
.
├── app.py              # Streamlit web application (entry point)
├── data.json           # Application metadata (input + saved results)
├── scanning.py         # OCR scanning logic (Tesseract + EasyOCR)
├── parsing.py          # Text matching & compliance checks
├── processing.py       # High-level label processing pipeline
├── utils.py            # Shared utilities (clustering, normalization, helpers)
├── cleaning.py         # Utility script to reset/clear previously processed results
├── test_labels/        # Label images (application_num.png)
```


## 5. Running the Application

From the project root directory, run:

```bash
streamlit run app.py
```

You should see output similar to:

```text
Local URL: http://localhost:8501
```

Open that URL in your browser.


## Assumptions

### Data
This implementation assumes information about an application is associated with each image of a label and accessible. As the specifics of the `.NET` framework are not known, I use a `.json` file to support storing relevant information about an application, such as the brand name and address of the brewer. This information is assumed to be easily accessible (if it can be included on a checklist for a compliance agent, it likely can be integrated with a lightweight local tool). Additionally, it assumes each application is complete without errors upon receiving it.


## Limitations

### Beverage class
This model is currently only able to process labels for domestic malt beverages.

### Qualitative Requirements 
There are requirements for labeling such as "does a distinctive or fanciful name together with an adequate and truthful statement of composition appear on the label?" This is not a clear yes or no question and requires discretion and experience to answer. While an advanced LLM likely could handle this question, the lack of clarity surrounding a general answer and the inefficiency of LLMs place questions like these outside the scope of the model. 

### Synthetic Data and Testing
While the model performs well on the small dataset (only 1 misclassification in 6 compliant and 6 non-compliant labels), limited token access has meant limited testing data. 