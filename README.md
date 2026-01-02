#  College Academic Risk Prediction System

## Overview
The College Academic Risk Prediction System is an end-to-end machine learning project that predicts a college student’s final academic score and classifies their academic risk level (At Risk, Average, or Top Performer). The project demonstrates the complete machine learning lifecycle, from data preprocessing and exploratory analysis to model deployment using a Streamlit web application.

## Problem Statement
Early identification of academically at-risk students allows educational institutions to provide timely academic support. This project aims to predict final academic performance and classify students into risk categories using historical academic data and machine learning techniques.

## Solution Approach
The project follows a structured machine learning workflow that includes data cleaning and preprocessing, exploratory data analysis, feature engineering, model training and evaluation, hyperparameter tuning using GridSearchCV, model explainability, and deployment using Streamlit.

## Project Structure
College_Academic_Risk_Prediction/
├── notebooks/
│ ├── 01_Data_Loading_and_Cleaning.ipynb
│ ├── 02_EDA_College_Academic_Risk.ipynb
│ ├── 03_Modeling_and_Tuning.ipynb
│ └── 04_Explainability_and_Results.ipynb
├── app/
│ └── app.py
├── data/
│ └── processed/
│ └── student_featured.csv
├── models/
│ ├── final_score_model.pkl
│ └── risk_classifier.pkl
├── README.md
└── requirements.txt

## Dataset Description
The dataset consists of anonymized academic information of college students, including attendance percentage, internal assessment marks, daily self-study hours, backlog count, and final academic score. Additional engineered features such as performance index and consistency score were created to better capture academic behavior.

## Exploratory Data Analysis
Exploratory data analysis was conducted to understand the distribution of academic scores and risk categories, identify key predictors of academic risk, validate engineered features, and analyze non-linear relationships between variables. The analysis revealed that attendance, internal marks, and backlog count strongly influence academic performance.

## Feature Engineering
Custom features were engineered to improve predictive power and interpretability. Performance Index combines attendance and internal performance, while Consistency Score measures stability across internal assessments. These features enhanced both regression and classification performance.

## Models Used
For final score prediction, a Random Forest Regressor was used. For academic risk classification, a Random Forest Classifier was implemented. Random Forest models were chosen due to their ability to handle non-linear relationships, robustness to outliers, and inherent feature importance capabilities.

## Model Optimization and Evaluation
Hyperparameter tuning was performed using GridSearchCV with cross-validation. Regression performance was evaluated using RMSE, while classification performance was assessed using accuracy and classification reports.

## Explainability
Model explainability was achieved through feature importance analysis and risk-wise comparison of key features. This ensures transparency and helps understand which factors most influence academic performance and risk.

## Web Application
An interactive Streamlit web application was developed to allow users to input student academic details, predict final academic score, classify academic risk level, and view explanations and recommendations. The application is suitable for local execution as well as cloud deployment.

## How to Run the Application Locally
Install dependencies using `pip install -r requirements.txt`. Navigate to the app directory and run `streamlit run app.py`. Open the application in a browser at `http://localhost:8501`.

## Deployment
The application is deployment-ready and can be hosted on Streamlit Cloud using `app/app.py` as the entry point.

## Interview Highlights
This project demonstrates building a complete machine learning pipeline, applying feature engineering and hyperparameter tuning, combining regression and classification in a single system, focusing on explainability, and deploying a real-world machine learning application.

## Future Improvements
Future enhancements include SHAP-based explainability, database integration, automated retraining pipelines, and institution-level analytics dashboards.

## Author
Mansi Rajput  

Machine Learning and Data Science Enthusiast
