# 🏡 EstateIQ — Intelligent Real Estate Price Prediction & Market Analytics

### Turning Property Data into Smarter Real Estate Decisions

EstateIQ is an end-to-end Machine Learning and Real Estate Analytics platform designed to estimate residential property listing prices across India. It combines data science, predictive modelling, and interactive analytics to transform property listing data into meaningful market insights.

Built using Python, Scikit-learn, XGBoost, and Streamlit, EstateIQ enables users to explore real estate trends, compare property markets, and estimate property prices based on key property characteristics.

---

## 🌟 Project Highlights

| Feature | Description |
|---|---|
| 🏠 Price Prediction | Estimate residential property listing prices |
| 📊 Market Analytics | Explore property pricing patterns and trends |
| 📍 Location Intelligence | Compare real estate markets across Indian locations |
| 🤖 Machine Learning | Compare multiple regression algorithms |
| 📈 Model Evaluation | Analyse MAE, RMSE, and R² performance |
| 🖥️ Interactive Dashboard | Explore insights through a Streamlit application |

## 🎯 Business Problem

The Indian real estate market is diverse, with property prices varying significantly across locations, property sizes, furnishing conditions, ownership types, and other characteristics.

Understanding these variations can be challenging for buyers, sellers, and real estate analysts.

EstateIQ addresses this problem by using historical property listing data and machine learning techniques to:

- Estimate property listing prices.
- Identify important property characteristics associated with pricing.
- Compare listing-price trends across locations.
- Support data-driven real estate market exploration.

## 📊 Dataset Overview

**Dataset:** [House Price (India) — Kaggle](https://www.kaggle.com/datasets/juhibhojani/house-price)

**Dataset Author:** Juhi Bhojani

**License:** Community Data License Agreement – Sharing 1.0 (CDLA-Sharing-1.0)

| Dataset Information | Details |
|---|---|
| Original Records | 187,531 |
| Cleaned Records | 177,071 |
| Locations Covered | 81 |
| Domain | Indian Residential Real Estate |
| Prediction Target | Property Listing Price |

### Features Used

- Location
- BHK
- Bathrooms and balconies
- Carpet area and super area
- Furnishing status
- Transaction type
- Ownership
- Property facing and overlooking
- Floor number and total floors
- Parking availability and type

## ⚙️ Machine Learning Workflow

The project follows a structured end-to-end machine learning pipeline.

**01 — Data Collection & Understanding**
- Imported and inspected the original property dataset.
- Examined feature distributions, data types, and data quality.

**02 — Data Cleaning**
- Handled missing values and inconsistent records.
- Investigated duplicate and extreme-value records.
- Prepared the cleaned dataset for analysis.

**03 — Exploratory Data Analysis**
- Analysed property prices across locations.
- Explored BHK-wise and furnishing-wise price patterns.
- Investigated relationships between carpet area, bathrooms, and listing prices.

**04 — Feature Engineering & Preprocessing**
- Applied numerical imputation and StandardScaler.
- Used categorical imputation and OneHotEncoder.
- Applied `log1p` transformation to the target variable.
- Split the dataset into 80% training and 20% testing data.

**05 — Model Development**
Three regression algorithms were developed and evaluated:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

**06 — Model Evaluation**
Models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

**07 — Application Development**
Developed an interactive Streamlit application integrating the trained model and preprocessing pipeline.

## 🤖 Model Performance

The following results were obtained on the held-out test dataset. Error metrics are reported in Indian Rupees.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Random Forest | ₹10.70 Lakh | ₹52.23 Lakh | 0.8522 |
| XGBoost | ₹13.59 Lakh | ₹53.68 Lakh | 0.8440 |
| Linear Regression | ₹81.67 Lakh | ₹3,829.58 Lakh | -793.2440 |

**Selected Model: Random Forest Regressor**

Random Forest was selected for the initial application because it achieved the lowest MAE and strongest R² score on the original price scale among the evaluated models.

## 🛠️ Technology Stack

| Category | Tools & Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Feature Engineering | StandardScaler, OneHotEncoder |
| Model Serialization | Joblib |
| Application Development | Streamlit |
| Version Control | Git, GitHub |

## 🖥️ EstateIQ Application

The Streamlit application brings together multiple functionalities in one interface.

### 1. Overview Dashboard
Explore key dataset statistics, property listing distributions, and market-level insights.

### 2. Price Predictor
Enter property characteristics and receive an estimated listing price using the trained Random Forest model.

### 3. Market Analytics
Explore price patterns across locations, BHK configurations, furnishing categories, and property areas.

### 4. Model Insights
Compare the performance of different machine learning algorithms using evaluation metrics and visualizations.

### 5. About Project
Understand the project objectives, dataset, and machine learning methodology.

## 🚀 Installation & Execution

### Prerequisites
- Python 3.10 or later
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/EstateIQ-House-Price-Prediction-Copy.git
```

### Step 2 — Navigate to the project

```bash
cd EstateIQ-House-Price-Prediction-Copy
```

### Step 3 — Create a virtual environment

```bash
python -m venv .venv
```

### Step 4 — Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Run the application

```bash
streamlit run app.py
```

**Note:** The application also requires the trained model and preprocessing files (`random_forest.pkl` and `preprocessor.pkl`) in `data/models/`, along with the processed dataset files used by the dashboard. These artifacts must be obtained separately if they are not included in the repository.

## 📁 Project Structure

```text
EstateIQ/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── processed/
│   └── models/
│
├── notebooks/
│   └── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── clean_data.py
│   └── predict.py
│
└── models/
```

## 🔮 Future Enhancements

- Explainable AI integration using SHAP.
- Advanced hyperparameter optimization.
- More granular location-level market intelligence.
- Integration of updated real estate listing datasets.
- Model monitoring and periodic retraining.
- Cloud deployment and improved application accessibility.

## ⚠️ Disclaimer

EstateIQ is an educational Machine Learning and Analytics project. The dataset contains property listing and asking-price information rather than verified completed sale transactions.

Predictions are estimates based on historical listing data and should not be considered guaranteed selling prices, professional property valuations, or financial advice.

## 👩‍💻 Author

### Zeba Sayyed

**Data Science & Analytics | Machine Learning | Generative AI**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/zebasayyed17)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/zeba-sayyed-28766941b/)

---

⭐ If you find this project interesting, consider giving the repository a star!

*Built with Python, Machine Learning, and a passion for data-driven innovation.*
