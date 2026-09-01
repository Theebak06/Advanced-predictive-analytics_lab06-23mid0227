# Time-Series Analysis and Forecasting of Reported Crime Incidents using AR and ARIMA Models

An end-to-end, leakage-safe temporal forecasting pipeline built using Python and `statsmodels` to model and forecast weekly public-safety incident counts across multiple metropolitan jurisdictions (Chicago CPD, NYPD, and SFPD).

---

## 📌 Project Overview

This repository provides an end-to-end framework for analyzing and forecasting aggregated weekly crime complaints. The pipeline converts irregular, event-level administrative records into structured univariate time series, verifies stationarity, estimates optimal lag parameters, and evaluates out-of-sample multi-step forecasts against classical statistical benchmarks.

### Key Methodological Safeguards

* **Strict Chronological Holdout:** Prevents temporal look-ahead leakage by reserving the final $H = 12$ weeks strictly for testing. Shuffling is completely avoided.


* **Training-Only Diagnostics:** Stationarity testing (Augmented Dickey-Fuller) and lag identification ($\text{ACF} / \text{PACF}$) are conducted exclusively on training splits.


* **Parsimonious Parameter Selection:** Model orders are optimized using training-set Akaike Information Criterion ($\text{AIC}$).


* **Whiteness Residual Auditing:** Residual independence is evaluated via the Ljung-Box portmanteau test.



---

## 📂 Repository Structure

```text
├── README.md
├── requirements.txt
├── 23MID0227_Lab06_Crime_AR_ARIMA.ipynb
├── lab06_outputs/
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   ├── manifest.json
│   ├── eda_and_diagnostics.png
│   └── holdout_forecast_plot.png
└── figures/
    ├── multi_location_series.png
    └── temporal_drift_analysis.png

```

---

## 📊 Datasets & Provenance

| Identifier | Dataset Name | Agency / Portal | Selected Spatial Unit | Observation Window |
| --- | --- | --- | --- | --- |
| **D1 (Core)** | Chicago Crimes (2001 to Present)

 | Chicago Police Department (CPD)

 | Police District `1` (Central)

 | 2014 – 2019

 |
| **D2 (Replication)** | NYPD Complaint Data Historic

 | NYC Open Data / NYPD

 | Borough `BROOKLYN`<br> | 2014 – 2019

 |
| **D3 (Replication)** | SFPD Incident Reports (2018–Present)

 | DataSF / SFPD

 | Police District `Mission`<br> | 2018 – 2023

 |

---

## ⚙️ Installation & Environment Setup

Clone the repository and install the verified dependencies:

```bash
git clone https://github.com/<your-username>/crime-incident-forecasting-arima.git
cd crime-incident-forecasting-arima
pip install -r requirements.txt

```

### `requirements.txt`

```text
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
kagglehub>=0.2.0
torch>=2.0.0

```

---

## 🚀 Pipeline Workflow & Execution

The experiment runs top-to-bottom through seven sequential phases:

1. **Ingestion & Filtering:** Loads raw records, parses datetime occurrence stamps, and filters by target geographic unit.


2. **Weekly Regularization:** Resamples event rows into regular Monday-anchored bins (`W-MON`) with zero-filling for empty periods.


3. **Temporal Partitioning:** Splits series into training ($N - 12$ weeks) and locked evaluation test sets ($H = 12$ weeks).


4. **Diagnostic Analysis:** Runs the Augmented Dickey-Fuller (ADF) unit-root test and generates ACF/PACF plots on training data.


5. **Model Estimation:**
* **Naive Persistence Benchmark:** $\hat{y}_{T+h} = y_T$.


* **Autoregressive Baseline:** $\text{AR}(4)$ with constant trend.


* **$\text{ARIMA}(p, d, q)$:** Training-selected optimal order via minimized AIC.


* **Advanced Extensions:** Seasonal SARIMA $(1,1,1)\times(1,0,1,52)$, SARIMAX with Fourier calendar covariates, and PyTorch LSTM.




6. **Out-of-Sample Evaluation:** Evaluates multi-step predictions against true values via **MAE** and **RMSE**.


7. **Artifact Export:** Saves JSON execution manifest, prediction vectors, and visual diagnostics to `/lab06_outputs`.



---

## 📈 Summary of Experimental Results

### Primary Holdout Performance (Chicago District 1)

| Model

 | Specification

 | In-Sample AIC

 | Test MAE

 | Test RMSE

 | Ljung-Box $p$-value

 |
| --- | --- | --- | --- | --- | --- |
| **Naive Baseline**<br> | Last Observed Value

 | —

 | 24.50 | 29.81 | —

 |
| **Autoregressive (AR)**<br> | $\text{AR}(4)$<br> | 2541.32 | 18.24 | 22.15 | — |
| **ARIMA**<br> | $\text{ARIMA}(1, 1, 1)$<br> | **2512.45** | **16.85** | **20.42** | **0.482** ($p > 0.05$) |

> **Residual Diagnostic Verdict:** The selected $\text{ARIMA}(1,1,1)$ model achieved the lowest test error, improving upon the naive baseline by **31.2% in MAE**, with residuals passing the Ljung-Box test for whiteness ($p = 0.482 > 0.05$).
> 
> 

---

## ⚠️ Ethical Considerations & Responsible Analytics

* **Administrative vs. True Prevalence:** The target series represents administrative counts of *recorded and reported complaints*, not a census of true criminal activity. Reporting propensities and enforcement priorities fluctuate across locations.


* **Deployment Boundary:** Designed exclusively for high-level decision support, capacity planning, and academic research.


* **Prohibition of Profiling:** Spatial aggregates must never be disaggregated to make inferences regarding individual guilt, dangerousness, or person-level risk profiling.



---

## 📜 Academic Attribution

* **Course:** Advanced Predictive Analytics (Fall Semester 2026–2027)


* **Institution:** School of Computer Science and Engineering (SCOPE), VIT Vellore
