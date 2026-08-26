# Spam-Shield-Enterprise-ML-Email-Threat-Filter
An enterprise desktop security tool built with Python, CustomTkinter, and Scikit-Learn  . Features automated dataset harmonization, multi-model ML benchmarking (Naive Bayes, SVM, Logistic Regression, Random Forest), real-time threat risk grading, and embedded performance analytics  .
# 🛡️ Spam-Shield-Enterprise-ML-Email-Threat-Filter

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![GUI Framework](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![ML Engine](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)

An enterprise desktop security tool built with Python, CustomTkinter, and Scikit-Learn. Features automated dataset harmonization, multi-model ML benchmarking (Naive Bayes, SVM, Logistic Regression, Random Forest), real-time threat risk grading, and embedded performance analytics.

---

## 🌟 Key Features

* **Multi-Model Tournament Training**: Evaluates **Multinomial Naive Bayes**, **Logistic Regression**, **Linear SVM**, and **Random Forest**, automatically deploying the best model based on F1-score.
* **Intelligent Data Preprocessor**: Ingests diverse CSV structures, maps column variations (`text`, `message`, `body` vs. `label`, `spam`, `category`), strips noise (URLs, emails, punctuation), and handles deduplication.
* **Real-Time Risk Grading**: Provides class predictions along with calibrated confidence percentages and tiered threat levels (**Safe / Low**, **Moderate**, **High**, **Critical Risk**).
* **Embedded Analytics Suite**: Integrated Matplotlib visualizations featuring dataset distribution donut charts, model comparison matrices, and confusion heatmaps.
* **Modern Dark Glassmorphism UI**: Built on CustomTkinter with asynchronous training threads to maintain a smooth 60 FPS desktop experience.
* **Synthetic Fallback Dataset**: Automatically generates synthetic sample data if no local CSV files are found.

---

## 🏗️ ML & Architecture Pipeline

```text
Raw Email / CSV Input
         │
         ▼
┌─────────────────────────┐
│   NLP Preprocessing     │ ──> Lowercasing, regex URL/email stripping, deduplication
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   TF-IDF Vectorizer     │ ──> Unigram/Bigram feature extraction (Top 5,000 features)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Model Benchmarking    │ ──> Naive Bayes | Logistic Reg | Linear SVM | Random Forest
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Optimal Model Selector │ ──> Auto-export best model artifacts to ./models/ (F1-score)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│    Desktop Interface    │ ──> Real-time inference HUD with confidence & risk breakdown
└─────────────────────────┘
```
---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Spam-Shield-Enterprise-ML-Email-Threat-Filter.git](https://github.com/your-username/Spam-Shield-Enterprise-ML-Email-Threat-Filter.git)
cd Spam-Shield-Enterprise-ML-Email-Threat-Filter
