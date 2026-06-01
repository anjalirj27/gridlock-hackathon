# gridlock-hackathon
# Traffic Demand Prediction

## Overview

This project was developed as part of the **Gridlock Hackathon**. The objective is to predict traffic demand for road segments using spatial, road-network, and environmental features.

The dataset contains traffic-related information such as:

* Geohash (location identifier)
* Road Type
* Number of Lanes
* Temperature
* Day
* Traffic Demand (target variable)

The goal is to build machine learning models capable of accurately estimating normalized traffic demand values.

---

## Project Structure

```text
gridlock/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── processed/
│   │   ├── train_processed.csv
│   │   └── test_processed.csv
│   │
│   └── submissions/
│       ├── submission.csv
│       └── submission_YYYYMMDD_HHMMSS.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   ├── geohash_features.py
│   │   └── feature_builder.py
│   │
│   ├── models/
│   │   ├── train_catboost.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── config.py
│       └── helpers.py
│
├── reports/
│   ├── figures/
│   └── eda_summary.md
│
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

---

## Exploratory Data Analysis

Several exploratory analyses were performed to understand the dataset.

### Demand Distribution

Observations:

* Demand values are bounded between 0 and 1.
* The distribution is highly right-skewed.
* Skewness ≈ 3.7.
* Most observations have relatively low demand.

### Road Type Analysis

Road types are highly imbalanced:

| Road Type   |   Count |
| ----------- | ------: |
| Residential | ~69,230 |
| Street      |  ~3,909 |
| Highway     |  ~3,560 |

Demand is strongly associated with road type:

| Road Type   | Mean Demand |
| ----------- | ----------: |
| Residential |      ~0.057 |
| Street      |      ~0.273 |
| Highway     |      ~0.611 |

### Geohash Analysis

* Approximately 1,249 unique geohashes.
* Geohash exhibits a strong relationship with demand.
* Correlation Ratio (Eta) ≈ 0.83.

This indicates that spatial location is a major determinant of traffic demand.

### Latitude and Longitude

Geohashes were decoded into latitude and longitude coordinates.

Findings:

* Linear correlation between latitude/longitude and demand was very weak.
* Spatial information is still highly predictive.
* Demand appears to be driven by spatial clusters rather than simple linear geographic trends.

---

## Baseline Models

### Geohash Mean Lookup

Prediction:

```text
Demand = Mean demand of each geohash
```

RMSE ≈ 0.0786

### Road Type Mean Lookup

Prediction:

```text
Demand = Mean demand of each road type
```

RMSE ≈ 0.0704

### Geohash + Road Type Lookup

Prediction:

```text
Demand = Mean demand for each (RoadType, Geohash) pair
```

RMSE ≈ 0.0483

---

## Machine Learning Model

### CatBoost Regressor

Features:

* Geohash
* RoadType

Categorical handling:

* Native CatBoost categorical encoding

Performance:

```text
RMSE ≈ 0.0532
R²   ≈ 0.844
```

CatBoost was selected because:

* Excellent support for categorical features.
* Minimal preprocessing requirements.
* Strong performance on tabular datasets.

---

## Feature Engineering

Current features:

* Geohash
* RoadType
* Temperature
* NumberOfLanes
* Day

Additional experiments:

* Geohash decoding to latitude and longitude.
* Spatial feature analysis.
* Sequential residual modeling inspired by Gram-Schmidt orthogonalization.

---

## Running the Project

### Create Environment

```bash
conda create -n gridlock python=3.12
conda activate gridlock
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Train Model

```bash
python src/models/train_catboost.py
```

### Generate Submission

```bash
python src/models/predict.py
```

Submission files will be saved under:

```text
data/submissions/
```

---

## Future Improvements

* Geohash embeddings
* Target encoding
* Spatial clustering features
* CatBoost hyperparameter optimization
* Ensemble methods (CatBoost + LightGBM)
* Temporal feature engineering
* Graph-based road network features

---

## Author

Shivam Kumar

B.Tech Civil Engineering

Machine Learning and AI Enthusiast
