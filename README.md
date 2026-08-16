# 🔨 PrediKit

> Forge powerful machine learning models with ease - No coding required!

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2.3-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Models & Parameters](#models--parameters)
- [Tutorials](#tutorials)
  - [Tutorial 1: Iris Flower Classification](#-tutorial-1-iris-flower-classification)
  - [Tutorial 2: House Price Prediction](#-tutorial-2-house-price-prediction)
- [Understanding Results](#understanding-results)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

ML-Forge is a user-friendly web-based tool that democratizes machine learning by providing a visual interface for training, evaluating, and deploying ML models. Whether you're a data scientist prototyping solutions or a business analyst exploring predictive analytics, ML-Forge eliminates the coding barrier while maintaining professional-grade functionality.

### What Makes ML-Forge Special?

- **🚀 No Code Required**: Upload your data, click buttons, get results
- **🎯 Professional Quality**: Uses scikit-learn's battle-tested algorithms
- **📊 Interactive Visualizations**: Understand your model's performance at a glance
- **💾 Production Ready**: Export predictions with sample IDs for easy integration
- **📋 Three-Sheet Format**: Train, Validation, Test separation built-in
- **🔧 Smart Defaults**: Intelligent parameter handling for all models

## ✨ Key Features

### 📊 Data Management
- Upload Excel files with Train/Val/Test worksheets
- Automatic classification vs regression detection
- Handles categorical and numerical features
- Preserves sample IDs for traceability
- Works with or without test labels (uses validation if test is empty)
- Automatic handling of missing values (NaN)

### 🤖 Model Training
| Model | Type | Best For |
|-------|------|----------|
| **Decision Tree** | Interpretable | Understanding feature importance |
| **Random Forest** | Ensemble | High accuracy, robust performance |
| **SVM** | Kernel-based | High-dimensional data |
| **Neural Network (DNN)** | Deep Learning | Complex patterns, large datasets |

### 📈 Evaluation & Visualization
- **Classification**: Accuracy, F1-Score, Precision, Recall, Confusion Matrix
- **Regression**: R², RMSE, MAE, MSE, Line Plot, Scatter Plot, Residual Plot
- Interactive charts and graphs
- Exportable results with sample IDs

### 💾 Export Capabilities
- Download predictions with Sample IDs
- Excel format with summary sheets
- Includes actual values (when available)
- Tracks which dataset was used (Test or Validation)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ML-Forge.git
cd ML-Forge

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run ML-Forge
python app.py