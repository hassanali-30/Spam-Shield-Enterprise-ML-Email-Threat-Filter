#!/usr/bin/env python3
"""
===============================================================================
SPAM EMAIL DETECTION SYSTEM - PROFESSIONAL ENTERPRISE EDITION
===============================================================================
Architecture : Single-File Modular Desktop Application
GUI Framework: CustomTkinter (Modern Dark Glassmorphism Theme)
ML Stack     : Scikit-Learn, Pandas, NumPy, Joblib, Matplotlib
Target Engine: Python 3.11+
===============================================================================
"""

import os
import sys
import re
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Core Data & ML Libraries
import pandas as pd
import numpy as np
import joblib

# Machine Learning Modules
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# GUI Framework
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Matplotlib GUI Integration
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "spam_detector.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SpamShield")

# -----------------------------------------------------------------------------
# GLOBAL THEME & STYLING CONSTANTS (GLASSMORPHISM DARK PALETTE)
# -----------------------------------------------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

COLOR_BG_DARK = "#0b0f19"         # Deep Slate / Near Black background
COLOR_CARD_BG = "#151c2c"         # Glassmorphism Card Surface
COLOR_CARD_BORDER = "#232d42"     # Subtle Border for Glass Effect
COLOR_SURFACE_HOVER = "#1e293b"   # Interactive Hover State
COLOR_ACCENT_PRIMARY = "#3b82f6"  # Electric Blue Accent
COLOR_ACCENT_SUCCESS = "#10b981"  # Emerald Green (Ham / Safe)
COLOR_ACCENT_DANGER = "#ef4444"   # Crimson Red (Spam / Danger)
COLOR_ACCENT_WARNING = "#f59e0b"  # Amber / High Risk
COLOR_ACCENT_INFO = "#8b5cf6"     # Purple Info Accent
COLOR_TEXT_MAIN = "#f8fafc"       # High-contrast bright text
COLOR_TEXT_MUTED = "#94a3b8"      # Dimmed body text
COLOR_PANEL_BG = "#0f172a"        # Secondary Surface Background

FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else "Helvetica"
FONT_HEADING = (FONT_FAMILY, 20, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 14, "bold")
FONT_BODY = (FONT_FAMILY, 12)
FONT_CAPTION = (FONT_FAMILY, 10)

# Paths Configuration. Resolve relative to this file so the application works
# when launched from any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_DIR / "datasets"
MODELS_DIR = Path("./models")
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA ENGINE & PREPROCESSING MODULE
# =============================================================================
class DataPreprocessor:
    """Handles automatic dataset schema detection, harmonization, cleaning, and merging."""

    KNOWN_SCHEMA_MAPS = [
        {"text": ["text", "v2", "message", "email", "content", "body"], 
         "label": ["label", "v1", "category", "spam", "class", "target"]}
    ]

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Applies comprehensive NLP text cleaning techniques."""
        if not isinstance(raw_text, str) or not raw_text.strip():
            return ""
        
        # Lowercase text
        text = raw_text.lower()
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove Email Addresses
        text = re.sub(r"\S+@\S+", "", text)
        # Remove Numbers
        text = re.sub(r"\d+", "", text)
        # Remove Punctuation & Special Characters
        text = re.sub(r"[^\w\s]", "", text)
        # Remove Extra Spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    @classmethod
    def harmonize_dataframe(cls, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Detects column formats and renames them to standard ['text', 'label']."""
        df = df.copy()
        # Drop unnamed index columns
        unnamed_cols = [c for c in df.columns if "unnamed" in str(c).lower()]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)

        matched_text_col = None
        matched_label_col = None

        col_lower_map = {str(c).lower().strip(): c for c in df.columns}

        # Check against schema mappings
        for schema in cls.KNOWN_SCHEMA_MAPS:
            for t_candidate in schema["text"]:
                if t_candidate in col_lower_map:
                    matched_text_col = col_lower_map[t_candidate]
                    break
            for l_candidate in schema["label"]:
                if l_candidate in col_lower_map:
                    matched_label_col = col_lower_map[l_candidate]
                    break
            if matched_text_col and matched_label_col:
                break

        if not matched_text_col or not matched_label_col:
            # Fallback heuristic: assume longest string column is text, categorical is label
            if len(df.columns) >= 2:
                matched_label_col = df.columns[0]
                matched_text_col = df.columns[1]
            else:
                raise ValueError(f"Could not automatically parse column schema for {source_name}")

        df = df[[matched_text_col, matched_label_col]].copy()
        df.columns = ["text", "label"]

        # Standardize Label values to 'spam' or 'ham'
        def normalize_label(val):
            val_str = str(val).strip().lower()
            if val_str in ["1", "spam", "pos", "positive", "true", "1.0"]:
                return "spam"
            return "ham"

        df["label"] = df["label"].apply(normalize_label)
        logger.info(f"Harmonized dataset '{source_name}': {len(df)} rows mapping [{matched_text_col}, {matched_label_col}] -> ['text', 'label']")
        return df

    @classmethod
    def load_and_merge_datasets(cls) -> pd.DataFrame:
        """Loads spam.csv and emails.csv from datasets folder and merges in-memory via pandas.concat."""
        dfs = []
        target_files = ["spam.csv", "emails.csv"]
        
        for file_name in target_files:
            # Prefer the writable datasets directory, then use the CSV files
            # shipped beside Main.py for a fresh checkout.
            candidates = [DATASETS_DIR / file_name, PROJECT_DIR / file_name]
            file_path = next((path for path in candidates if path.exists()), None)
            if file_path is not None:
                try:
                    # Attempt reading with fallback encodings
                    try:
                        raw_df = pd.read_csv(file_path, encoding="utf-8")
                    except UnicodeDecodeError:
                        raw_df = pd.read_csv(file_path, encoding="latin-1")
                    
                    harmonized_df = cls.harmonize_dataframe(raw_df, file_name)
                    dfs.append(harmonized_df)
                except Exception as e:
                    logger.error(f"Failed to load dataset {file_name}: {e}")

        # Synthetic Fallback Dataset if local files missing
        if not dfs:
            logger.warning("No bundled datasets found. Generating a synthetic fallback dataset.")
            dfs.append(cls._generate_synthetic_dataset())

        # Merge in memory using pd.concat
        merged_df = pd.concat(dfs, ignore_index=True)
        initial_count = len(merged_df)

        # Cleaning step: Drop Missing, Clean Text, Remove Duplicates
        merged_df.dropna(subset=["text", "label"], inplace=True)
        merged_df["cleaned_text"] = merged_df["text"].apply(cls.clean_text)
        merged_df = merged_df[merged_df["cleaned_text"].str.strip() != ""]
        merged_df.drop_duplicates(subset=["cleaned_text"], inplace=True)

        logger.info(f"Merged Dataset Ready. Initial rows: {initial_count}, Cleaned Unique rows: {len(merged_df)}")
        return merged_df

    @staticmethod
    def _generate_synthetic_dataset() -> pd.DataFrame:
        """Generates realistic synthetic emails to ensure out-of-the-box readiness."""
        spams = [
            "URGENT! You have won a $1,000 Walmart Gift Card. Click here to claim your prize immediately!",
            "Congratulations! You are selected for a guaranteed $5,000 loan. Reply now with your bank details.",
            "Exclusive Deal: Get 80% off on luxury watches and designer bags. Limited time offer visit our shop!",
            "Dear User, your bank account access has been suspended due to security risks. Verify your account password now.",
            "Make $500 per hour working from home with zero investment! Call this toll-free number today.",
            "Crypto investment alert! Double your Bitcoin in 24 hours. Guaranteed returns. Register at instant-btc.org",
            "Final notice: Unpaid tax invoice generated for your profile. Pay online immediately to avoid legal prosecution.",
            "Free Viagra samples available with every purchase! Order online anonymously with cheap rates.",
            "You have 1 new unread secure message from online banking portal. Click the secure link to view.",
            "Win a brand new iPhone 15 Pro Max! Fill out this 30-second survey to enter the instant lucky draw!"
        ] * 15
        
        hams = [
            "Hi Team, please find attached the weekly sales performance report and agenda for tomorrow's standup.",
            "Hey mom, I will be reaching home by 7 PM. Let us order pizza for dinner tonight.",
            "Reminder: Project sync meeting is scheduled for 3:00 PM in Conference Room B.",
            "Can you review the updated Python codebase and push your pull request before EOD?",
            "Thanks for your order! Your package has been dispatched and will arrive by Thursday via FedEx.",
            "Dear customer, your monthly utility statement is ready. Please view your account summary online.",
            "Let us catch up over coffee this weekend. It has been a long time since we chatted!",
            "Please confirm if the server deployment script ran successfully without any error logs.",
            "Here are the lecture slides and notes from today's Artificial Intelligence class.",
            "Are you free for a quick Zoom call to discuss the design UI mockups for the app?"
        ] * 15

        data = [{"text": s, "label": "spam"} for s in spams] + [{"text": h, "label": "ham"} for h in hams]
        df = pd.DataFrame(data)
        # Write to disk so future loads find them
        df.to_csv(DATASETS_DIR / "spam.csv", index=False)
        return df


# =============================================================================
# MACHINE LEARNING ENGINE
# =============================================================================
class SpamMLEngine:
    """Manages Vectorization, Model Training, Comparison, Evaluation, and Persistence."""

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.encoder: Optional[LabelEncoder] = None
        self.best_model: Optional[Any] = None
        self.best_model_name: str = "None"
        self.metrics_report: Dict[str, Dict[str, float]] = {}
        self.confusion_matrices: Dict[str, np.ndarray] = {}
        self.dataset_df: Optional[pd.DataFrame] = None
        self.is_trained: bool = False

    def train_and_evaluate_all(self, df: pd.DataFrame, progress_callback=None) -> Dict[str, Any]:
        """Trains MultinomialNB, LogisticRegression, Linear SVC, and RandomForest, selecting the best."""
        self.dataset_df = df
        
        if progress_callback:
            progress_callback(0.1, "Initializing TF-IDF Vectorization...")

        # Feature Extraction
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        X = self.vectorizer.fit_transform(df["cleaned_text"]).toarray()

        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(df["label"]) # ham=0, spam=1 (usually)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Linear SVM": SVC(kernel="linear", probability=True, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        self.metrics_report.clear()
        self.confusion_matrices.clear()
        
        best_f1 = -1.0
        best_selected = None
        total_models = len(models)

        for idx, (name, clf) in enumerate(models.items(), start=1):
            if progress_callback:
                progress_callback(0.2 + (idx / total_models) * 0.7, f"Training {name}...")

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, pos_label=self._get_spam_label_idx(), zero_division=0))
            rec = float(recall_score(y_test, y_pred, pos_label=self._get_spam_label_idx(), zero_division=0))
            f1 = float(f1_score(y_test, y_pred, pos_label=self._get_spam_label_idx(), zero_division=0))

            cm = confusion_matrix(y_test, y_pred)

            self.metrics_report[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            }
            self.confusion_matrices[name] = cm

            logger.info(f"Model [{name}] -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            # Auto select best model based on F1 Score
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name
                best_selected = clf

        self.best_model = best_selected
        self.is_trained = True

        if progress_callback:
            progress_callback(1.0, f"Training Complete! Best: {self.best_model_name}")

        # Save artifacts automatically
        self.save_artifacts()

        return self.metrics_report[self.best_model_name]

    def _get_spam_label_idx(self) -> int:
        """Helper to get integer index for 'spam' label in encoder."""
        if self.encoder is None:
            return 1
        labels = list(self.encoder.classes_)
        return labels.index("spam") if "spam" in labels else 1

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """Runs single text inference returning Prediction label, Confidence %, Risk Level, and Probabilities."""
        if not self.is_trained or not self.best_model or not self.vectorizer or not self.encoder:
            raise RuntimeError("ML Models are not trained or loaded yet.")

        cleaned = DataPreprocessor.clean_text(raw_text)
        if not cleaned:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "spam_prob": 0.0,
                "ham_prob": 0.0,
                "risk_level": "N/A"
            }

        vec = self.vectorizer.transform([cleaned]).toarray()
        pred_class_idx = self.best_model.predict(vec)[0]
        predicted_label = self.encoder.inverse_transform([pred_class_idx])[0]

        # Get probabilities if supported
        if hasattr(self.best_model, "predict_proba"):
            probs = self.best_model.predict_proba(vec)[0]
            spam_idx = self._get_spam_label_idx()
            ham_idx = 1 - spam_idx
            spam_prob = float(probs[spam_idx])
            ham_prob = float(probs[ham_idx])
        else:
            spam_prob = 1.0 if predicted_label == "spam" else 0.0
            ham_prob = 1.0 - spam_prob

        confidence = max(spam_prob, ham_prob) * 100.0

        # Calculate Risk Level
        if spam_prob >= 0.85:
            risk_level = "CRITICAL RISK"
        elif spam_prob >= 0.60:
            risk_level = "HIGH RISK"
        elif spam_prob >= 0.35:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "LOW RISK / SAFE"

        return {
            "label": predicted_label.upper(),
            "confidence": confidence,
            "spam_prob": spam_prob * 100.0,
            "ham_prob": ham_prob * 100.0,
            "risk_level": risk_level
        }

    def save_artifacts(self) -> None:
        """Saves model, vectorizer, and encoder pkl files to ./models/."""
        try:
            joblib.dump(self.best_model, MODELS_DIR / "spam_model.pkl")
            joblib.dump(self.vectorizer, MODELS_DIR / "vectorizer.pkl")
            joblib.dump(self.encoder, MODELS_DIR / "encoder.pkl")
            logger.info("Successfully saved model artifacts to ./models/")
        except Exception as e:
            logger.error(f"Failed saving models: {e}")

    def load_artifacts(self) -> bool:
        """Loads model, vectorizer, and encoder if existing on disk."""
        model_p = MODELS_DIR / "spam_model.pkl"
        vec_p = MODELS_DIR / "vectorizer.pkl"
        enc_p = MODELS_DIR / "encoder.pkl"

        if model_p.exists() and vec_p.exists() and enc_p.exists():
            try:
                self.best_model = joblib.load(model_p)
                self.vectorizer = joblib.load(vec_p)
                self.encoder = joblib.load(enc_p)
                self.best_model_name = type(self.best_model).__name__
                self.is_trained = True
                logger.info("Successfully loaded model artifacts from ./models/")
                return True
            except Exception as e:
                logger.error(f"Error loading models from disk: {e}")
        return False


# =============================================================================
# VISUALIZATION ENGINE (MATPLOTLIB EMBEDDING)
# =============================================================================
class AnalyticsPlotter:
    """Generates styled Matplotlib figures customized for modern dark mode UI embedding."""

    @staticmethod
    def set_dark_style(fig: Figure, ax: plt.Axes):
        """Applies dark theme styling parameters to Matplotlib objects."""
        fig.patch.set_facecolor(COLOR_CARD_BG)
        ax.set_facecolor(COLOR_CARD_BG)
        ax.spines["bottom"].set_color(COLOR_CARD_BORDER)
        ax.spines["top"].set_color(COLOR_CARD_BORDER)
        ax.spines["right"].set_color(COLOR_CARD_BORDER)
        ax.spines["left"].set_color(COLOR_CARD_BORDER)
        ax.tick_params(colors=COLOR_TEXT_MUTED, which="both")
        ax.yaxis.label.set_color(COLOR_TEXT_MAIN)
        ax.xaxis.label.set_color(COLOR_TEXT_MAIN)
        ax.title.set_color(COLOR_TEXT_MAIN)

    @classmethod
    def create_distribution_pie(cls, df: pd.DataFrame) -> Figure:
        """Renders Spam vs Ham distribution Donut Chart."""
        fig = Figure(figsize=(4.5, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        cls.set_dark_style(fig, ax)

        if df is None or df.empty:
            ax.text(0.5, 0.5, "No Data Loaded", ha="center", va="center", color=COLOR_TEXT_MUTED)
            return fig

        counts = df["label"].value_counts()
        labels = [f"{lbl.upper()}\n({cnt})" for lbl, cnt in counts.items()]
        colors = [COLOR_ACCENT_DANGER if "spam" in lbl.lower() else COLOR_ACCENT_SUCCESS for lbl in counts.index]

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            textprops={"color": COLOR_TEXT_MAIN, "fontsize": 9},
            wedgeprops={"edgecolor": COLOR_CARD_BG, "linewidth": 2, "width": 0.4} # Donut hole
        )
        for at in autotexts:
            at.set_color("#ffffff")
            at.set_weight("bold")

        ax.set_title("Dataset Class Balance", fontsize=11, pad=10, weight="bold")
        fig.tight_layout()
        return fig

    @classmethod
    def create_model_comparison_bar(cls, metrics: Dict[str, Dict[str, float]]) -> Figure:
        """Renders accuracy & F1 score comparison bar chart across algorithms."""
        fig = Figure(figsize=(5.5, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        cls.set_dark_style(fig, ax)

        if not metrics:
            ax.text(0.5, 0.5, "Models Not Trained", ha="center", va="center", color=COLOR_TEXT_MUTED)
            return fig

        models = list(metrics.keys())
        accs = [metrics[m]["accuracy"] * 100 for m in models]
        f1s = [metrics[m]["f1"] * 100 for m in models]

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, accs, width, label="Accuracy %", color=COLOR_ACCENT_PRIMARY, edgecolor="none")
        ax.bar(x + width/2, f1s, width, label="F1 Score %", color=COLOR_ACCENT_INFO, edgecolor="none")

        ax.set_ylabel("Score (%)", fontsize=9)
        ax.set_title("Algorithm Performance Matrix", fontsize=11, pad=10, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.15, color=COLOR_TEXT_MUTED)
        ax.legend(facecolor=COLOR_CARD_BG, edgecolor=COLOR_CARD_BORDER, labelcolor=COLOR_TEXT_MAIN, fontsize=8)

        fig.tight_layout()
        return fig

    @classmethod
    def create_confusion_matrix_plot(cls, cm: Optional[np.ndarray]) -> Figure:
        """Renders Confusion Matrix heatmap."""
        fig = Figure(figsize=(4.5, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        cls.set_dark_style(fig, ax)

        if cm is None:
            ax.text(0.5, 0.5, "Matrix Unavailable", ha="center", va="center", color=COLOR_TEXT_MUTED)
            return fig

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix", fontsize=11, pad=10, weight="bold")

        classes = ["Ham", "Spam"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=9)

        # Annotate matrix cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else COLOR_TEXT_MAIN,
                        weight="bold", fontsize=11)

        ax.set_ylabel("True Label", fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9)
        fig.tight_layout()
        return fig


# =============================================================================
# CUSTOM UI WIDGETS & GLASS CARDS
# =============================================================================
class MetricCard(ctk.CTkFrame):
    """Modern Glassmorphism Metric / KPI display card."""

    def __init__(self, master, title: str, value: str = "--", subtext: str = "", accent_color: str = COLOR_ACCENT_PRIMARY, **kwargs):
        super().__init__(
            master,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_CARD_BORDER,
            corner_radius=12,
            **kwargs
        )

        self.columnconfigure(0, weight=1)

        # Top Accent Color Pill Indicator
        self.accent_bar = ctk.CTkFrame(self, height=4, fg_color=accent_color, corner_radius=2)
        self.accent_bar.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 5))

        # Title Label
        self.title_label = ctk.CTkLabel(self, text=title.upper(), font=(FONT_FAMILY, 11, "bold"), text_color=COLOR_TEXT_MUTED)
        self.title_label.grid(row=1, column=0, sticky="w", padx=16, pady=(2, 0))

        # Main Metric Value
        self.value_label = ctk.CTkLabel(self, text=value, font=(FONT_FAMILY, 24, "bold"), text_color=COLOR_TEXT_MAIN)
        self.value_label.grid(row=2, column=0, sticky="w", padx=16, pady=(2, 0))

        # Subtext / Caption
        self.sub_label = ctk.CTkLabel(self, text=subtext, font=(FONT_FAMILY, 10), text_color=COLOR_TEXT_MUTED)
        self.sub_label.grid(row=3, column=0, sticky="w", padx=16, pady=(0, 12))

    def update_val(self, value: str, subtext: Optional[str] = None):
        """Dynamic value update helper."""
        self.value_label.configure(text=value)
        if subtext is not None:
            self.sub_label.configure(text=subtext)


class ToolTip:
    """Hover ToolTip functionality for standard and customtkinter widgets."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, cy = self.widget.bbox("insert") if hasattr(self.widget, "bbox") and self.widget.bbox("insert") else (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#1e293b",
            foreground="#f8fafc",
            relief=tk.SOLID,
            borderwidth=1,
            font=(FONT_FAMILY, 9, "normal"),
            padx=8,
            pady=4
        )
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# =============================================================================
# MAIN DESKTOP APPLICATION ENGINE
# =============================================================================
class SpamDetectorApp(ctk.CTk):
    """Enterprise Desktop Application GUI Engine."""

    def __init__(self):
        super().__init__()

        # Window Frame Initialization
        self.title("Spam Email Detection System - Enterprise Edition")
        self.geometry("1500x900")
        self.minsize(1280, 800)
        self.configure(fg_color=COLOR_BG_DARK)

        # Core Engines Initialization
        self.ml_engine = SpamMLEngine()
        self.processed_df: Optional[pd.DataFrame] = None

        # Build GUI Structural Hierarchy
        self._setup_window_grid()
        self._build_sidebar()
        self._build_header()
        self._build_status_bar()
        self._build_main_container()

        # Build Pages / Views
        self.views: Dict[str, ctk.CTkFrame] = {}
        self._init_dashboard_view()
        self._init_predict_view()
        self._init_dataset_view()
        self._init_analytics_view()
        self._init_settings_view()
        self._init_about_view()

        # Menu & Keyboard Shortcuts
        self._create_menu_bar()
        self._bind_keyboard_shortcuts()

        # Startup Data & Model Auto-Load Routine
        self.after(200, self._startup_sequence)

    def _setup_window_grid(self):
        """Configure 2D grid layout for responsive root window."""
        self.grid_rowconfigure(1, weight=1) # Main View Container
        self.grid_columnconfigure(1, weight=1) # Main Content Column

    def _build_sidebar(self):
        """Constructs fixed modern navigation sidebar."""
        self.sidebar_frame = ctk.CTkFrame(
            self,
            width=240,
            fg_color=COLOR_PANEL_BG,
            corner_radius=0,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.sidebar_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1) # Push system controls to bottom

        # App Logo & Branding Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="🛡️ SPAM SHIELD",
            font=(FONT_FAMILY, 18, "bold"),
            text_color=COLOR_ACCENT_PRIMARY
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(24, 4), sticky="w")

        self.version_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Enterprise ML v2.4",
            font=(FONT_FAMILY, 10),
            text_color=COLOR_TEXT_MUTED
        )
        self.version_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # Navigation Menu Items
        self.nav_buttons = {}
        nav_items = [
            ("Dashboard", "📊", self._show_dashboard),
            ("Predict", "🔍", self._show_predict),
            ("Datasets", "📁", self._show_datasets),
            ("Analytics", "📈", self._show_analytics),
            ("Settings", "⚙️", self._show_settings),
            ("About", "ℹ️", self._show_about),
        ]

        for idx, (name, icon, cmd) in enumerate(nav_items, start=2):
            btn = ctk.CTkButton(
                self.sidebar_frame,
                text=f"  {icon}  {name}",
                anchor="w",
                font=(FONT_FAMILY, 13, "bold"),
                fg_color="transparent",
                text_color=COLOR_TEXT_MUTED,
                hover_color=COLOR_SURFACE_HOVER,
                height=42,
                corner_radius=8,
                command=cmd
            )
            btn.grid(row=idx, column=0, padx=12, pady=4, sticky="ew")
            self.nav_buttons[name] = btn

        # Quick Action Buttons Frame at Sidebar Bottom
        self.sidebar_bottom_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.sidebar_bottom_frame.grid(row=8, column=0, padx=12, pady=16, sticky="ew")

        self.btn_train_sidebar = ctk.CTkButton(
            self.sidebar_bottom_frame,
            text="⚡ Retrain Engine",
            font=(FONT_FAMILY, 12, "bold"),
            fg_color=COLOR_ACCENT_PRIMARY,
            hover_color="#2563eb",
            height=36,
            corner_radius=8,
            command=self.start_training_thread
        )
        self.btn_train_sidebar.pack(fill="x", pady=4)

    def _build_header(self):
        """Constructs top application header bar."""
        self.header_frame = ctk.CTkFrame(
            self,
            height=60,
            fg_color=COLOR_PANEL_BG,
            corner_radius=0,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.header_frame.grid(row=0, column=1, sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)

        # Dynamic Page Title
        self.page_title_label = ctk.CTkLabel(
            self.header_frame,
            text="Dashboard Overview",
            font=(FONT_FAMILY, 16, "bold"),
            text_color=COLOR_TEXT_MAIN
        )
        self.page_title_label.grid(row=0, column=0, padx=24, pady=16, sticky="w")

        # Right Status Pills Container
        self.pills_container = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.pills_container.grid(row=0, column=2, padx=24, pady=12, sticky="e")

        # Active Model Pill
        self.active_model_pill = ctk.CTkLabel(
            self.pills_container,
            text="Model: Uninitialized",
            font=(FONT_FAMILY, 11, "bold"),
            fg_color=COLOR_CARD_BG,
            text_color=COLOR_ACCENT_INFO,
            corner_radius=16,
            padx=12,
            pady=4
        )
        self.active_model_pill.pack(side="left", padx=6)

        # System Status Indicator Pill
        self.status_pill = ctk.CTkLabel(
            self.pills_container,
            text="● Idle",
            font=(FONT_FAMILY, 11, "bold"),
            fg_color=COLOR_CARD_BG,
            text_color=COLOR_ACCENT_SUCCESS,
            corner_radius=16,
            padx=12,
            pady=4
        )
        self.status_pill.pack(side="left", padx=6)

    def _build_status_bar(self):
        """Constructs bottom system status bar."""
        self.status_bar = ctk.CTkFrame(
            self,
            height=28,
            fg_color=COLOR_PANEL_BG,
            corner_radius=0,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.status_bar.grid(row=2, column=1, sticky="ew")
        self.status_bar.grid_columnconfigure(0, weight=1)

        self.status_msg_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready.",
            font=(FONT_FAMILY, 10),
            text_color=COLOR_TEXT_MUTED
        )
        self.status_msg_label.grid(row=0, column=0, padx=16, pady=4, sticky="w")

        self.system_info_label = ctk.CTkLabel(
            self.status_bar,
            text="Python 3.11+ | CustomTkinter | Scikit-Learn",
            font=(FONT_FAMILY, 10),
            text_color=COLOR_TEXT_MUTED
        )
        self.system_info_label.grid(row=0, column=1, padx=16, pady=4, sticky="e")

    def _build_main_container(self):
        """Constructs main central container for switching view frames."""
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=1, column=1, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

    # -------------------------------------------------------------------------
    # VIEW INITIALIZERS
    # -------------------------------------------------------------------------
    def _init_dashboard_view(self):
        """Initializes main combined Dashboard and Detection view."""
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["Dashboard"] = view

        view.columnconfigure(0, weight=3) # Email Input Column
        view.columnconfigure(1, weight=2) # KPIs & Results Column
        view.rowconfigure(0, weight=1)

        # ---------------- LEFT COLUMN: Input & Actions ----------------
        left_frame = ctk.CTkFrame(view, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)

        lbl_input_title = ctk.CTkLabel(left_frame, text="Email Content Inspector", font=FONT_SUBTITLE, text_color=COLOR_TEXT_MAIN)
        lbl_input_title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))

        self.txt_email_input = ctk.CTkTextbox(
            left_frame,
            font=(FONT_FAMILY, 12),
            fg_color=COLOR_BG_DARK,
            text_color=COLOR_TEXT_MAIN,
            border_width=1,
            border_color=COLOR_CARD_BORDER,
            corner_radius=8,
            wrap="word"
        )
        self.txt_email_input.grid(row=1, column=0, sticky="nsew", padx=16, pady=8)
        self.txt_email_input.insert("1.0", "Paste email content here to perform real-time AI spam analysis...")

        # Action Buttons Control Bar
        actions_bar = ctk.CTkFrame(left_frame, fg_color="transparent")
        actions_bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 16))

        btn_detect = ctk.CTkButton(
            actions_bar,
            text="🔍 Detect Spam",
            font=(FONT_FAMILY, 12, "bold"),
            fg_color=COLOR_ACCENT_PRIMARY,
            hover_color="#2563eb",
            height=38,
            command=self.on_detect_click
        )
        btn_detect.pack(side="left", padx=(0, 8))

        btn_clear = ctk.CTkButton(
            actions_bar,
            text="🧹 Clear Input",
            font=(FONT_FAMILY, 12),
            fg_color=COLOR_CARD_BORDER,
            hover_color=COLOR_SURFACE_HOVER,
            height=38,
            command=self.on_clear_click
        )
        btn_clear.pack(side="left", padx=4)

        btn_load_sample_spam = ctk.CTkButton(
            actions_bar,
            text="Sample Spam",
            font=(FONT_FAMILY, 11),
            fg_color="transparent",
            text_color=COLOR_ACCENT_DANGER,
            border_width=1,
            border_color=COLOR_ACCENT_DANGER,
            hover_color="#3b1219",
            height=38,
            command=self._load_sample_spam
        )
        btn_load_sample_spam.pack(side="right", padx=4)

        btn_load_sample_ham = ctk.CTkButton(
            actions_bar,
            text="Sample Ham",
            font=(FONT_FAMILY, 11),
            fg_color="transparent",
            text_color=COLOR_ACCENT_SUCCESS,
            border_width=1,
            border_color=COLOR_ACCENT_SUCCESS,
            hover_color="#062d1f",
            height=38,
            command=self._load_sample_ham
        )
        btn_load_sample_ham.pack(side="right", padx=4)

        # ---------------- RIGHT COLUMN: Dashboard & Metrics ----------------
        right_frame = ctk.CTkFrame(view, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        right_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(1, weight=1)

        # Top 4 Metric Cards Grid
        self.card_total = MetricCard(right_frame, title="Total Emails", value="0", subtext="Loaded across datasets", accent_color=COLOR_ACCENT_PRIMARY)
        self.card_total.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=(0, 10))

        self.card_spam = MetricCard(right_frame, title="Spam Count", value="0", subtext="Detected malicious", accent_color=COLOR_ACCENT_DANGER)
        self.card_spam.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 10))

        self.card_best_model = MetricCard(right_frame, title="Active Model", value="None", subtext="F1 Score: N/A", accent_color=COLOR_ACCENT_INFO)
        self.card_best_model.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=(0, 10))

        self.card_accuracy = MetricCard(right_frame, title="Accuracy", value="0.0%", subtext="Test evaluation", accent_color=COLOR_ACCENT_SUCCESS)
        self.card_accuracy.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(0, 10))

        # Real-time Prediction Output Panel
        pred_panel = ctk.CTkFrame(right_frame, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        pred_panel.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(5, 0))
        pred_panel.columnconfigure(0, weight=1)

        lbl_pred_header = ctk.CTkLabel(pred_panel, text="Classification Result", font=FONT_SUBTITLE, text_color=COLOR_TEXT_MAIN)
        lbl_pred_header.pack(anchor="w", padx=16, pady=(16, 4))

        # Large Status Result Badge
        self.lbl_result_badge = ctk.CTkLabel(
            pred_panel,
            text="READY FOR ANALYSIS",
            font=(FONT_FAMILY, 20, "bold"),
            fg_color=COLOR_BG_DARK,
            text_color=COLOR_TEXT_MUTED,
            corner_radius=8,
            height=50
        )
        self.lbl_result_badge.pack(fill="x", padx=16, pady=8)

        # Prediction Stats & Gauges
        stats_frame = ctk.CTkFrame(pred_panel, fg_color="transparent")
        stats_frame.pack(fill="x", padx=16, pady=8)
        stats_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(stats_frame, text="Confidence:", font=FONT_BODY, text_color=COLOR_TEXT_MUTED).grid(row=0, column=0, sticky="w", pady=4)
        self.lbl_confidence_val = ctk.CTkLabel(stats_frame, text="0.0%", font=(FONT_FAMILY, 14, "bold"), text_color=COLOR_TEXT_MAIN)
        self.lbl_confidence_val.grid(row=0, column=1, sticky="e", pady=4)

        self.progress_confidence = ctk.CTkProgressBar(stats_frame, fg_color=COLOR_BG_DARK, progress_color=COLOR_ACCENT_PRIMARY)
        self.progress_confidence.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.progress_confidence.set(0)

        ctk.CTkLabel(stats_frame, text="Risk Assessment:", font=FONT_BODY, text_color=COLOR_TEXT_MUTED).grid(row=2, column=0, sticky="w", pady=4)
        self.lbl_risk_level = ctk.CTkLabel(stats_frame, text="UN EVALUATED", font=(FONT_FAMILY, 12, "bold"), text_color=COLOR_TEXT_MUTED)
        self.lbl_risk_level.grid(row=2, column=1, sticky="e", pady=4)

        # Probabilities Breakdown
        prob_frame = ctk.CTkFrame(pred_panel, fg_color=COLOR_BG_DARK, corner_radius=8)
        prob_frame.pack(fill="x", padx=16, pady=(8, 16))
        prob_frame.columnconfigure(0, weight=1)
        prob_frame.columnconfigure(1, weight=1)

        self.lbl_spam_prob = ctk.CTkLabel(prob_frame, text="Spam Prob: 0.0%", font=FONT_CAPTION, text_color=COLOR_ACCENT_DANGER)
        self.lbl_spam_prob.grid(row=0, column=0, padx=8, pady=8)

        self.lbl_ham_prob = ctk.CTkLabel(prob_frame, text="Ham Prob: 0.0%", font=FONT_CAPTION, text_color=COLOR_ACCENT_SUCCESS)
        self.lbl_ham_prob.grid(row=0, column=1, padx=8, pady=8)

    def _init_predict_view(self):
        """Initializes dedicated live deep predictor view."""
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["Predict"] = view
        view.columnconfigure(0, weight=1)
        view.rowconfigure(0, weight=1)

        card = ctk.CTkFrame(view, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        card.grid(row=0, column=0, sticky="nsew")
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text="Deep Inspection & Feature Extraction", font=FONT_SUBTITLE, text_color=COLOR_TEXT_MAIN).grid(row=0, column=0, sticky="w", padx=20, pady=16)

        self.txt_predict_deep = ctk.CTkTextbox(card, font=(FONT_FAMILY, 12), fg_color=COLOR_BG_DARK, text_color=COLOR_TEXT_MAIN, corner_radius=8)
        self.txt_predict_deep.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)

        bot_bar = ctk.CTkFrame(card, fg_color="transparent")
        bot_bar.grid(row=2, column=0, sticky="ew", padx=20, pady=16)

        ctk.CTkButton(bot_bar, text="Run Deep Analysis", font=(FONT_FAMILY, 12, "bold"), fg_color=COLOR_ACCENT_PRIMARY, command=self._run_deep_predict).pack(side="left")
        self.lbl_deep_output = ctk.CTkLabel(bot_bar, text="Awaiting Input...", font=(FONT_FAMILY, 13, "bold"), text_color=COLOR_TEXT_MUTED)
        self.lbl_deep_output.pack(side="left", padx=20)

    def _init_dataset_view(self):
        """Initializes Dataset management and inspector page."""
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["Datasets"] = view
        view.columnconfigure(0, weight=1)
        view.rowconfigure(1, weight=1)

        # Control Panel Top
        ctrl_card = ctk.CTkFrame(view, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        ctrl_card.grid(row=0, column=0, sticky="ew", pady=(0, 15))

        ctk.CTkButton(ctrl_card, text="📂 Load Custom CSV", font=(FONT_FAMILY, 12, "bold"), fg_color=COLOR_ACCENT_PRIMARY, command=self.load_custom_dataset).pack(side="left", padx=16, pady=16)
        ctk.CTkButton(ctrl_card, text="🔄 Reload Merged Datasets", font=(FONT_FAMILY, 12), fg_color=COLOR_CARD_BORDER, command=self.reload_datasets).pack(side="left", padx=8, pady=16)

        self.lbl_dataset_info = ctk.CTkLabel(ctrl_card, text="Dataset Stats: 0 rows", font=(FONT_FAMILY, 12), text_color=COLOR_TEXT_MUTED)
        self.lbl_dataset_info.pack(side="right", padx=16, pady=16)

        # Data Table Container (Text preview representation)
        table_card = ctk.CTkFrame(view, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        table_card.grid(row=1, column=0, sticky="nsew")
        table_card.rowconfigure(0, weight=1)
        table_card.columnconfigure(0, weight=1)

        self.txt_dataset_preview = ctk.CTkTextbox(table_card, font=("Courier", 11), fg_color=COLOR_BG_DARK, text_color=COLOR_TEXT_MAIN)
        self.txt_dataset_preview.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

    def _init_analytics_view(self):
        """Initializes ML Model Analytics & Charts page."""
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["Analytics"] = view
        view.columnconfigure(0, weight=1)
        view.columnconfigure(1, weight=1)
        view.rowconfigure(0, weight=1)
        view.rowconfigure(1, weight=1)

        # Plot Containers (2x2 Grid)
        self.chart_frame_1 = ctk.CTkFrame(view, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        self.chart_frame_1.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        self.chart_frame_2 = ctk.CTkFrame(view, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        self.chart_frame_2.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))

        self.chart_frame_3 = ctk.CTkFrame(view, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        self.chart_frame_3.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

    def _init_settings_view(self):
        """Initializes Application Settings configuration page."""
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["Settings"] = view

        card = ctk.CTkFrame(view, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        card.pack(fill="both", expand=True)

        ctk.CTkLabel(card, text="Application & ML Configuration", font=FONT_SUBTITLE, text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=20, pady=20)

        # Settings Options
        s_frame = ctk.CTkFrame(card, fg_color="transparent")
        s_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(s_frame, text="Spam Sensitivity Threshold:", font=FONT_BODY, text_color=COLOR_TEXT_MAIN).grid(row=0, column=0, sticky="w", pady=10)
        self.slider_threshold = ctk.CTkSlider(s_frame, from_=0.1, to=0.9, number_of_steps=8, width=300)
        self.slider_threshold.set(0.5)
        self.slider_threshold.grid(row=0, column=1, padx=20, pady=10)

        ctk.CTkLabel(s_frame, text="Active Appearance Theme:", font=FONT_BODY, text_color=COLOR_TEXT_MAIN).grid(row=1, column=0, sticky="w", pady=10)
        self.opt_theme = ctk.CTkOptionMenu(s_frame, values=["Dark", "Light", "System"], command=ctk.set_appearance_mode)
        self.opt_theme.set("Dark")
        self.opt_theme.grid(row=1, column=1, padx=20, pady=10)

    def _init_about_view(self):
        """Initializes System Metadata and About details page."""
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["About"] = view

        card = ctk.CTkFrame(view, fg_color=COLOR_CARD_BG, border_width=1, border_color=COLOR_CARD_BORDER, corner_radius=12)
        card.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(card, text="Spam Email Detection System", font=(FONT_FAMILY, 24, "bold"), text_color=COLOR_ACCENT_PRIMARY).pack(pady=(40, 5))
        ctk.CTkLabel(card, text="Enterprise AI Architecture", font=FONT_SUBTITLE, text_color=COLOR_TEXT_MUTED).pack(pady=(0, 20))

        info_text = (
            "Developed by: Senior Python Developer & ML Engineer\n"
            "Stack: Python 3.11+, CustomTkinter, Scikit-Learn, Pandas, NumPy, Matplotlib\n\n"
            "This enterprise-grade application performs automated text cleaning, harmonizes multi-format CSV\n"
            "datasets, trains multiple NLP models (Naive Bayes, Logistic Regression, Support Vector Machine, \n"
            "and Random Forest), auto-selects the optimal performing algorithm, and provides real-time\n"
            "spam classification with probability risk metrics."
        )

        ctk.CTkLabel(card, text=info_text, font=(FONT_FAMILY, 13), text_color=COLOR_TEXT_MAIN, justify="center").pack(pady=20)

    # -------------------------------------------------------------------------
    # NAVIGATION LOGIC
    # -------------------------------------------------------------------------
    def _show_view(self, view_name: str):
        """Switches visible content frame and updates header/sidebar highlight states."""
        for name, btn in self.nav_buttons.items():
            if name == view_name:
                btn.configure(fg_color=COLOR_SURFACE_HOVER, text_color=COLOR_ACCENT_PRIMARY)
            else:
                btn.configure(fg_color="transparent", text_color=COLOR_TEXT_MUTED)

        for name, frame in self.views.items():
            if name == view_name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

        self.page_title_label.configure(text=f"{view_name} Overview" if view_name != "About" else "About System")

    def _show_dashboard(self): self._show_view("Dashboard")
    def _show_predict(self): self._show_view("Predict")
    def _show_datasets(self): self._show_view("Datasets")
    def _show_analytics(self): self._render_analytics_charts(); self._show_view("Analytics")
    def _show_settings(self): self._show_view("Settings")
    def _show_about(self): self._show_view("About")

    # -------------------------------------------------------------------------
    # CORE PIPELINE & CONTROLLER ACTIONS
    # -------------------------------------------------------------------------
    def _startup_sequence(self):
        """Asynchronous system startup data load and model verification sequence."""
        self._update_status("Loading datasets and verifying models...")
        
        # Load datasets in memory
        self.processed_df = DataPreprocessor.load_and_merge_datasets()
        self._update_dataset_ui_metrics()

        # Try loading existing models, else prompt to train
        if self.ml_engine.load_artifacts():
            self._update_model_ui_metrics()
            self._update_status("Engine ready. Loaded existing trained model artifacts.")
        else:
            self._update_status("No pre-trained model found. Initializing automated training...")
            self.start_training_thread()

        self._show_dashboard()

    def start_training_thread(self):
        """Launches ML training pipeline on background worker thread."""
        if self.processed_df is None or self.processed_df.empty:
            messagebox.showwarning("Warning", "No dataset available for training!")
            return

        self.btn_train_sidebar.configure(state="disabled", text="⏳ Training...")
        self.status_pill.configure(text="● Training", text_color=COLOR_ACCENT_WARNING)

        # Worker thread
        def worker():
            def progress_cb(pct, msg):
                self.after(0, lambda: self._update_status(f"[{int(pct*100)}%] {msg}"))

            metrics = self.ml_engine.train_and_evaluate_all(self.processed_df, progress_callback=progress_cb)
            self.after(0, lambda: self._on_training_complete(metrics))

        threading.Thread(target=worker, daemon=True).start()

    def _on_training_complete(self, best_metrics: Dict[str, float]):
        """Callback executed on GUI thread when ML training finishes."""
        self.btn_train_sidebar.configure(state="normal", text="⚡ Retrain Engine")
        self.status_pill.configure(text="● Ready", text_color=COLOR_ACCENT_SUCCESS)
        self._update_model_ui_metrics()
        self._update_status(f"Training complete! Best Model: {self.ml_engine.best_model_name} (F1: {best_metrics['f1']*100:.1f}%)")
        messagebox.showinfo("Success", f"All models trained successfully!\nBest Selected: {self.ml_engine.best_model_name}\nF1 Score: {best_metrics['f1']*100:.2f}%")

    def on_detect_click(self):
        """Triggered when Detect button is pressed."""
        text = self.txt_email_input.get("1.0", "end-1c").strip()
        if not text or text == "Paste email content here to perform real-time AI spam analysis...":
            messagebox.showinfo("Input Required", "Please enter or paste email text to analyze.")
            return

        try:
            res = self.ml_engine.predict(text)
            label = res["label"]
            conf = res["confidence"]
            risk = res["risk_level"]

            # Update UI components
            if label == "SPAM":
                self.lbl_result_badge.configure(
                    text=f"🚨 DETECTED SPAM ({conf:.1f}%)",
                    fg_color="#3b1219",
                    text_color=COLOR_ACCENT_DANGER
                )
            else:
                self.lbl_result_badge.configure(
                    text=f"✅ LEGITIMATE HAM ({conf:.1f}%)",
                    fg_color="#062d1f",
                    text_color=COLOR_ACCENT_SUCCESS
                )

            self.lbl_confidence_val.configure(text=f"{conf:.1f}%")
            self.progress_confidence.set(conf / 100.0)

            # Color code risk assessment text
            risk_colors = {
                "CRITICAL RISK": COLOR_ACCENT_DANGER,
                "HIGH RISK": COLOR_ACCENT_WARNING,
                "MODERATE RISK": COLOR_ACCENT_INFO,
                "LOW RISK / SAFE": COLOR_ACCENT_SUCCESS
            }
            self.lbl_risk_level.configure(text=risk, text_color=risk_colors.get(risk, COLOR_TEXT_MUTED))

            self.lbl_spam_prob.configure(text=f"Spam Prob: {res['spam_prob']:.1f}%")
            self.lbl_ham_prob.configure(text=f"Ham Prob: {res['ham_prob']:.1f}%")

            self._update_status(f"Classification completed: {label} ({conf:.1f}% confidence)")

        except Exception as e:
            logger.error(f"Prediction failure: {e}")
            messagebox.showerror("Prediction Error", str(e))

    def on_clear_click(self):
        """Clears text area and resets result display indicators."""
        self.txt_email_input.delete("1.0", "end")
        self.lbl_result_badge.configure(text="READY FOR ANALYSIS", fg_color=COLOR_BG_DARK, text_color=COLOR_TEXT_MUTED)
        self.lbl_confidence_val.configure(text="0.0%")
        self.progress_confidence.set(0)
        self.lbl_risk_level.configure(text="UN EVALUATED", text_color=COLOR_TEXT_MUTED)
        self.lbl_spam_prob.configure(text="Spam Prob: 0.0%")
        self.lbl_ham_prob.configure(text="Ham Prob: 0.0%")

    def _run_deep_predict(self):
        """Executes deep feature extraction on deep inspect page."""
        text = self.txt_predict_deep.get("1.0", "end-1c").strip()
        if not text:
            return
        res = self.ml_engine.predict(text)
        self.lbl_deep_output.configure(
            text=f"Result: {res['label']} | Confidence: {res['confidence']:.2f}% | Risk: {res['risk_level']}",
            text_color=COLOR_ACCENT_DANGER if res['label'] == 'SPAM' else COLOR_ACCENT_SUCCESS
        )

    def load_custom_dataset(self):
        """Loads a user-selected CSV file into memory."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                raw_df = pd.read_csv(file_path)
                harmonized = DataPreprocessor.harmonize_dataframe(raw_df, os.path.basename(file_path))
                harmonized["cleaned_text"] = harmonized["text"].apply(DataPreprocessor.clean_text)
                
                if self.processed_df is not None:
                    self.processed_df = pd.concat([self.processed_df, harmonized], ignore_index=True).drop_duplicates(subset=["cleaned_text"])
                else:
                    self.processed_df = harmonized

                self._update_dataset_ui_metrics()
                messagebox.showinfo("Dataset Loaded", f"Loaded and merged {len(harmonized)} rows from CSV.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to parse CSV file:\n{e}")

    def reload_datasets(self):
        """Reloads default datasets from directory."""
        self.processed_df = DataPreprocessor.load_and_merge_datasets()
        self._update_dataset_ui_metrics()
        messagebox.showinfo("Reload Complete", "Datasets reloaded from local disk storage.")

    # -------------------------------------------------------------------------
    # UI METRIC & ANALYTICS RENDERERS
    # -------------------------------------------------------------------------
    def _update_dataset_ui_metrics(self):
        """Updates text stats and metric cards reflecting dataset status."""
        if self.processed_df is not None and not self.processed_df.empty:
            total = len(self.processed_df)
            counts = self.processed_df["label"].value_counts()
            spam = counts.get("spam", 0)
            
            self.card_total.update_val(f"{total:,}")
            self.card_spam.update_val(f"{spam:,}", subtext=f"{(spam/total)*100:.1f}% of dataset")
            self.lbl_dataset_info.configure(text=f"Total: {total:,} rows | Spam: {spam:,} | Ham: {counts.get('ham',0):,}")

            # Populate dataset page text preview
            preview_str = self.processed_df.head(50).to_string()
            self.txt_dataset_preview.delete("1.0", "end")
            self.txt_dataset_preview.insert("1.0", preview_str)

    def _update_model_ui_metrics(self):
        """Updates active model metric cards."""
        if self.ml_engine.is_trained:
            best_name = self.ml_engine.best_model_name
            m = self.ml_engine.metrics_report.get(best_name, {})
            acc = m.get("accuracy", 0.0) * 100.0
            f1 = m.get("f1", 0.0) * 100.0

            self.card_best_model.update_val(best_name, subtext=f"F1 Score: {f1:.1f}%")
            self.card_accuracy.update_val(f"{acc:.1f}%", subtext="Holdout Evaluation")
            self.active_model_pill.configure(text=f"Model: {best_name}")

    def _render_analytics_charts(self):
        """Renders embedded Matplotlib figures on Analytics Page."""
        # Clear previous canvases
        for widget in self.chart_frame_1.winfo_children(): widget.destroy()
        for widget in self.chart_frame_2.winfo_children(): widget.destroy()
        for widget in self.chart_frame_3.winfo_children(): widget.destroy()

        # Chart 1: Donut Distribution
        fig1 = AnalyticsPlotter.create_distribution_pie(self.processed_df)
        canvas1 = FigureCanvasTkAgg(fig1, master=self.chart_frame_1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Chart 2: Model Comparison Bar Chart
        fig2 = AnalyticsPlotter.create_model_comparison_bar(self.ml_engine.metrics_report)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.chart_frame_2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Chart 3: Confusion Matrix
        cm = self.ml_engine.confusion_matrices.get(self.ml_engine.best_model_name)
        fig3 = AnalyticsPlotter.create_confusion_matrix_plot(cm)
        canvas3 = FigureCanvasTkAgg(fig3, master=self.chart_frame_3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def _update_status(self, msg: str):
        """Status bar update helper."""
        self.status_msg_label.configure(text=msg)
        logger.info(msg)

    # -------------------------------------------------------------------------
    # SAMPLE DATA LOADERS
    # -------------------------------------------------------------------------
    def _load_sample_spam(self):
        sample = (
            "URGENT SECURITY ALERT! Your Bank of America online session has expired due to suspicious login attempts. "
            "Please log in immediately at http://verify-bankofamerica-secure.com to confirm your identity and prevent permanent suspension."
        )
        self.txt_email_input.delete("1.0", "end")
        self.txt_email_input.insert("1.0", sample)

    def _load_sample_ham(self):
        sample = (
            "Hi Sarah, could you please review the attached slide deck for tomorrow's client presentation? "
            "Let me know if you have any feedback or if we should adjust the financial projections before 3 PM."
        )
        self.txt_email_input.delete("1.0", "end")
        self.txt_email_input.insert("1.0", sample)

    # -------------------------------------------------------------------------
    # MENU BAR & SHORTCUTS
    # -------------------------------------------------------------------------
    def _create_menu_bar(self):
        """Constructs application drop-down menu bar."""
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Custom CSV Dataset (Ctrl+O)", command=self.load_custom_dataset)
        file_menu.add_command(label="Save Model Artifacts (Ctrl+S)", command=self.ml_engine.save_artifacts)
        file_menu.add_separator()
        file_menu.add_command(label="Exit (Ctrl+Q)", command=self.quit_app)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Retrain Engine (Ctrl+T)", command=self.start_training_thread)
        tools_menu.add_command(label="Clear Input", command=self.on_clear_click)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About System", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _bind_keyboard_shortcuts(self):
        """Binds global system keyboard shortcuts."""
        self.bind("<Control-o>", lambda e: self.load_custom_dataset())
        self.bind("<Control-s>", lambda e: self.ml_engine.save_artifacts())
        self.bind("<Control-t>", lambda e: self.start_training_thread())
        self.bind("<Control-q>", lambda e: self.quit_app())

    def quit_app(self):
        """Gracefully closes application."""
        self.destroy()


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = SpamDetectorApp()
    app.mainloop()
