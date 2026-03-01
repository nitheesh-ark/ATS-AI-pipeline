#!/bin/bash
# install.sh — One-shot dependency installer for ATS Resume Screening System

set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ATS Resume System — Dependency Installer           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Detect OS and install Tesseract ───────────────────────────────────────────
echo "📦  Installing Tesseract OCR binary..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
    echo "✅  Tesseract installed (apt)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &>/dev/null; then
        brew install tesseract
        echo "✅  Tesseract installed (brew)"
    else
        echo "⚠️   Homebrew not found. Install Tesseract manually: https://brew.sh"
    fi
else
    echo "⚠️   Windows detected — install Tesseract manually:"
    echo "    https://github.com/UB-Mannheim/tesseract/wiki"
fi

# ── Python packages ───────────────────────────────────────────────────────────
echo ""
echo "📦  Installing Python packages..."
pip install \
    pdfplumber \
    "pdfminer.six" \
    PyMuPDF \
    pytesseract \
    Pillow \
    spacy \
    dateparser \
    rapidfuzz \
    sentence-transformers \
    scikit-learn \
    numpy \
    torch

# ── spaCy model ───────────────────────────────────────────────────────────────
echo ""
echo "📦  Downloading spaCy en_core_web_md model..."
python -m spacy download en_core_web_md

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅  All dependencies installed!                     ║"
echo "║                                                      ║"
echo "║  Run:  python main.py --resumes ./your_resumes/      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
