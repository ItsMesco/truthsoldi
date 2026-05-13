# 📈 TruthSoldi - Automated Trading Architecture

An experimental, fully automated algorithmic trading system consisting of two decoupled pipelines. This project focuses on the intersection of **Natural Language Processing (NLP)**, **Real-time Data Engineering**, and **Asynchronous Systems**.

## 🏗️ Architecture & System Design

The project is built with a focus on "Separation of Concerns" (SoC). Instead of a monolithic bot, it implements two isolated pipelines:

1.  **Financial News Pipeline (`v2/`)**:
    * Ingests high-frequency news streams via **Alpaca API**.
    * Uses a robust XGBoost model (`bot_brain_xgb_robust.pkl`) to identify trading signals based on traditional financial headlines.
2.  **Social & Geopolitical Pipeline (`real_bot/`)**:
    * Scrapes/Analyzes social media activity (e.g., "Truths" from the TruthSocial ecosystem).
    * Leverages **FinBERT** (a specialized BERT model for financial sentiment) to evaluate the market impact of political and social statements.

This decoupled architecture ensures **fault tolerance**: the failure of a specific social media scraper does not halt the financial news execution, and vice versa.

## ⚠️ Disclaimer (Software Engineering vs. Alpha)
**This repository is strictly for educational and architectural research purposes.**
Currently, the trading strategies operate with a **negative ROI**. The primary goal of this project was to solve complex engineering challenges, such as:
* Real-time data ingestion and normalization.
* Integrating Large Language Models (LLMs) into time-sensitive decision loops.
* Managing state and concurrency in Python.

**Do not use this bot with real funds** unless you are prepared to sponsor the global markets.

## 🚀 Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/tuo-username/truthsoldi.git](https://github.com/tuo-username/truthsoldi.git)
    cd truthsoldi
    ```

2.  **Environment Configuration**:
    Copy the template and fill in your Alpaca/News API credentials.
    ```bash
    cp .env.example .env
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r v2/requirements.txt
    ```

4.  **Model Weights (Large Files)**:
    Due to GitHub's 100MB file limit, the trained model weights are hosted in the GitHub Releases section of this repository. 
    * Download `model_weights.pt` and place it in `real_bot/finbert_trading_bot/`
    * Download `bot_brain_xgb_robust.pkl` and place it in `v2/`
    
    [📥 Download Models from Releases](https://github.com/ItsMesco/truthsoldi/releases/tag/v1.0-models)

## 🛠️ Tech Stack
* **Language**: Python 3.10+
* **ML/NLP**: FinBERT (HuggingFace Transformers), XGBoost, Scikit-learn
* **Data**: Pandas, Parquet (for efficient storage), YAML/JSON
* **Brokerage**: Alpaca API (Paper/Live)
