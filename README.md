# Bellabeat Advanced Health Data Analytics and Intelligence

## Repository Name

bellabeat-health-analytics-intelligence

## Repository Description

Enterprise-grade end-to-end health data analytics pipeline for the Bellabeat case study. This project performs large-scale data processing, anomaly detection, predictive forecasting, advanced statistical analysis, and executive-level visualization using Python. It generates over 30 professional analytical visualizations, correlation intelligence, anomaly insights, and an automated executive report.

---

## Overview

This repository contains a complete production-grade analytics system designed to analyze Bellabeat and Fitbit health tracker datasets. It processes large raw CSV datasets, performs automated cleaning and aggregation, generates advanced visualizations, detects anomalies, performs machine learning forecasting, and produces a fully automated executive report.

The system is designed to be memory-safe, scalable, and portfolio-ready.

---

## Key Features

### Data Processing

* Automated ingestion of multiple CSV files
* Duplicate removal and data aggregation
* Memory-efficient chunk processing
* Clean master dataset generation

### Statistical Analysis

* Pearson correlation matrix
* Spearman correlation matrix
* Kendall correlation matrix
* Covariance matrix
* Distribution analysis

### Visualization Intelligence

* 30+ professional visualizations
* Line charts
* Histograms
* Density plots
* Box plots
* PCA projection
* Forecast visualization
* Anomaly detection visualization

All visualizations include:

* Proper units
* Clear axis labels
* Reduced tick density
* No overlapping labels
* Dark professional theme

### Machine Learning

* Isolation Forest anomaly detection
* Random Forest predictive forecasting
* PCA dimensionality reduction

### Automated Reporting

* Executive PDF report generation
* Embedded visual intelligence
* Executive summary
* Professional layout

---

## Output Structure

```
Output/
│
├── charts/
│   ├── 30+ analytical visualizations
│   ├── correlation matrices
│   ├── anomaly detection chart
│   └── forecast visualization
│
├── data/
│   ├── master_aggregated.xlsx
│   └── correlation_matrix.xlsx
│
├── models/
│   └── forecast.xlsx
│
└── Executive_Report.pdf
```

---

## Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
ReportLab
TQDM

---

## Installation

Create virtual environment:

```
python -m venv ml311
```

Activate environment:

Windows:

```
ml311\Scripts\activate
```

Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn reportlab tqdm openpyxl
```

---

## Usage

Place all dataset CSV files inside:

```
Input_Data/
```

Run the script:

```
python Data_Analyser.py
```

Output will automatically generate in:

```
Output/
```

---

## Executive Intelligence Generated

The system automatically produces:

* Executive report
* Forecast intelligence
* Anomaly detection insights
* Correlation intelligence
* Statistical distributions
* Predictive modeling outputs

---

## Portfolio Value

This project demonstrates:

* Enterprise data analytics
* Machine learning forecasting
* Anomaly detection
* Automated reporting
* Professional visualization
* Large-scale data processing

Suitable for roles in:

* Data Analyst
* Data Scientist
* Machine Learning Engineer
* Business Intelligence Analyst

---

## Author

Sagnik Sen

---

## License

MIT License
