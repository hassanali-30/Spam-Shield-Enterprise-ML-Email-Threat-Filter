# Spam-Shield Email Threat Filter

A Python desktop prototype for classifying email text and presenting risk-oriented results. It uses CustomTkinter for the interface and scikit-learn for text preprocessing, model comparison, and classification.

## Highlights

- Accepts common email/text and label column variations from CSV datasets
- Cleans and deduplicates text during preprocessing
- Uses TF-IDF features for text representation
- Compares Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest models
- Displays predictions, confidence-style scores, risk categories, and evaluation visualizations
- Includes dataset distribution, model comparison, and confusion-matrix views
- Includes a synthetic fallback dataset when no local CSV input is available

## Processing Flow

```text
CSV email data
    -> column mapping and cleaning
    -> TF-IDF feature extraction
    -> model comparison
    -> selected model and evaluation views
    -> desktop prediction interface
```

## Technology Stack

- Python 3.11+
- CustomTkinter
- scikit-learn
- Matplotlib

## Quick Start

Create and activate a virtual environment, then install the dependencies required by the application:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project's dependencies if a requirements file is provided:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python Main.py
```

## Included Data

The repository contains CSV datasets for experimentation. Check dataset licensing and privacy requirements before redistributing or using additional email data.

## Project Structure

```text
Main.py       # Application entry point
spam.csv      # Spam dataset
emails.csv    # Email dataset
README.md     # Project documentation
LICENSE       # License information
```

## Limitations

This is a learning and demonstration project. Model scores depend on the dataset and evaluation procedure; predictions should not be treated as a standalone security decision.