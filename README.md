# Loan Prediction Model

## Project Overview
This project focuses on building a machine learning model to predict loan approvals using the "Loan Prediction" dataset. The model was developed as part of the "Feature Forge: Enhancing & Evaluating ML Models" project, aligning with guidelines provided by Takeo's Data Analytics & AI Bootcamp.

## Objectives
1. Implement feature engineering techniques to preprocess and enhance the dataset.
2. Train, evaluate, and tune a machine learning model for accurate loan predictions.
3. Conduct stress tests and ensure deployment readiness of the model.

## Dataset Description
The dataset contains information about loan applicants, including:
- **Numerical features**: Applicant income, loan amount, credit history, etc.
- **Categorical features**: Education, property area, self-employment status, etc.
- **Target variable**: Loan approval status (approved/not approved).

### Key Statistics
- Total rows: 614
- Total columns (after encoding): 632

## Steps Completed

### 1. Exploratory Data Analysis (EDA)
- Handled missing values by:
  - Replacing numerical missing values with the mean.
  - Filling categorical missing values with "Unknown."
- Analyzed distributions, correlations, and data patterns.
- Generated a correlation heatmap to identify relationships between numerical features.

### 2. Feature Engineering
- Applied one-hot encoding to categorical variables, resulting in 632 features.
- Ensured all features were scaled using `StandardScaler` to improve model performance.

### 3. Model Training and Tuning
- Built a **Random Forest Classifier** as the primary model.
- Evaluated the model using metrics such as accuracy, precision, recall, and F1-score.
- Tuned hyperparameters using `RandomizedSearchCV` for optimal performance.
- Conducted cross-validation with a mean score of **81.05%**, confirming model robustness.

### 4. Stress Testing
- Introduced noise to test data to simulate real-world variability.
- The model demonstrated consistent performance under noisy conditions:
  - **Accuracy**: 78.86%
  - **F1-Score (True)**: 85.87%

### 5. Deployment Readiness
- Saved the tuned model as `loan_prediction_model.pkl` for deployment.

## Model Performance
| Metric            | Value     |
|-------------------|-----------|
| Accuracy          | 78.86%    |
| Precision (True)  | 75.96%    |
| Recall (True)     | 98.75%    |
| F1-Score (True)   | 85.87%    |
| Cross-Validation  | 81.05%    |

## Key Features and Importance
The top features influencing predictions include:
- Applicant Income
- Loan Amount
- Credit History
- Property Area

## Files Included
- **Jupyter Notebook**: Contains all project steps, from EDA to stress testing.
- **Model File**: `loan_prediction_model.pkl`
- **README**: Overview and details of the project.

## How to Use
```bash
# Load the model
import joblib
model = joblib.load('loan_prediction_model.pkl')

# Prepare input data similar to the dataset format
new_data = [/* Add your data here */]

# Make predictions
predictions = model.predict(new_data)
print(predictions)
