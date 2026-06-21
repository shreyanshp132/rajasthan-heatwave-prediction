# Rajasthan Heatwave Prediction

## Overview
An end-to-end machine learning pipeline predicting heatwave occurrence in Rajasthan using 20 years of meteorological data (2006-2025). Built with a production-style sklearn Pipeline combining ColumnTransformer, SimpleImputer, and RandomForestClassifier.

## Problem Statement
Heatwaves are rare but high-impact extreme weather events. Early prediction enables timely public health advisories and disaster preparedness. This project frames heatwave detection as an imbalanced binary classification problem using daily meteorological features.

## Dataset
- **Source:** Rajasthan Heatwave Dataset (2006-2025)
- **Size:** ~22,000 rows, 26 columns
- **Coverage:** Pre-monsoon season only (March-June) — the primary heatwave window in Rajasthan
- **Target:** HEATWAVE (0/1) — heavily imbalanced, only ~5% positive class

## Key Findings from EDA
- **May is the peak heatwave month** — probability over 10x higher than April
- No clear year-over-year increasing trend — heatwave frequency is volatile (e.g. 2010 had 145 days, 2008 had ~2)
- Heatwave days are not purely defined by a hard TMAX threshold — there's overlap between classes, indicating additional criteria (consecutive days, deviation from normal) influence the official label

## Feature Engineering
- Combined YEAR/MONTH/DAY into a proper datetime column for chronological sorting
- Created **lag features** (previous day's TMAX) and **rolling averages** (3-day TMAX) to capture temperature persistence — a key driver of heatwave conditions

## Pipeline Architecture
```
ColumnTransformer
├── Numeric features → SimpleImputer(median) → StandardScaler
└── Categorical (DISTRICT) → SimpleImputer(most_frequent) → OneHotEncoder
        ↓
RandomForestClassifier(class_weight='balanced')
```
Single saved pipeline object — preprocessing and model travel together, ready for deployment.

## Results

| Metric | Class 0 (No Heatwave) | Class 1 (Heatwave) |
|--------|------------------------|---------------------|
| Precision | 1.00 | 0.97 |
| Recall | 1.00 | 0.93 |
| F1-Score | 1.00 | 0.95 |

**Overall Accuracy: 99%** — caught 204 of 220 actual heatwave days with only 6 false alarms.

## Feature Importance Insight
Contrary to expectation, **TMIN and TEMP2M ranked as the top predictors** — not raw same-day TMAX. This suggests the model captures heatwave persistence through elevated *minimum* (nighttime) temperatures combined with general air temperature trends, rather than simply learning the threshold used to define the label. This aligns with meteorological understanding that heatwaves are characterized by sustained warmth, not just daytime peaks.

## Libraries Used
pandas, numpy, matplotlib, seaborn, scikit-learn

## How to Run
1. Install: `pip install pandas numpy matplotlib seaborn scikit-learn`
2. Place `Rajasthan_Heatwave_2006_2025.csv` in the working directory
3. Run `heatwave_prediction.ipynb`
