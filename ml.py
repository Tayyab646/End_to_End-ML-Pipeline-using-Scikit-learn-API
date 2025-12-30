# ==========================================
# 1. INSTALL & IMPORT DEPENDENCIES
# ==========================================
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ==========================================
# 2. LOAD & CLEAN DATA
# ==========================================
# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Handle 'TotalCharges' column which contains empty strings
# We convert them to NaN and then drop those rows (usually ~11 rows)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Define target and features
# Drop customerID as it's just an identifier
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Identify numerical and categorical columns
# SeniorCitizen is 0/1, we can treat it as categorical or numeric.
# Here we treat it as numeric for simplicity or categorical if we want to encode it.
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

# ==========================================
# 3. CONSTRUCT THE PREPROCESSING PIPELINE
# ==========================================
# Pipeline for numerical data: Impute missing values then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical data: Impute missing then One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Combine both using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==========================================
# 4. DEFINE MAIN PIPELINE & HYPERPARAMETER TUNING
# ==========================================
# Create a base pipeline with a placeholder classifier
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Define parameter grid for GridSearchCV
# We check both Logistic Regression and Random Forest in one go
param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=1000, solver='liblinear')],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l1', 'l2']
    },
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting Grid Search for hyperparameter tuning...")
grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# ==========================================
# 5. EVALUATION & EXPORT
# ==========================================
print(f"\nBest Model: {grid_search.best_params_['classifier']}")
print(f"Best Parameters: {grid_search.best_params_}")

# Get the best estimator
best_pipeline = grid_search.best_estimator_

# Make predictions
y_pred = best_pipeline.predict(X_test)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Export the entire pipeline for deployment
joblib.dump(best_pipeline, 'telco_churn_pipeline.joblib')
print("\nPipeline exported successfully as 'telco_churn_pipeline.joblib'")
