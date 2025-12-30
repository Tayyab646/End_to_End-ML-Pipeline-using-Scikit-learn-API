# End_to_End-ML-Pipeline-using-Scikit-learn-API

## 📌 Project Overview

This project is **End-to-End ML Pipeline with Scikit-learn Pipeline API**.
The purpose of this project is to build a **reusable and production-ready machine learning pipeline** for predicting **customer churn**.

The pipeline handles data preprocessing, model training, hyperparameter tuning, and model export in a clean and automated way.

---

## 🎯 Objective

To design and implement a complete machine learning workflow using **Scikit-learn’s Pipeline API** that can predict whether a customer will churn or not.

The pipeline is built in a way that it can be easily reused in real-world production systems.

---

## 📂 Dataset

* **Telco Customer Churn Dataset**
* Contains customer information such as:

  * Demographics
  * Services used
  * Account details
* Target variable: **Customer Churn (Yes / No)**

---

## 🛠️ Project Tasks

The following steps are performed in this project:

* Data preprocessing using **Scikit-learn Pipeline**

  * Feature scaling
  * Categorical encoding
* Training multiple machine learning models:

  * Logistic Regression
  * Random Forest
* Hyperparameter tuning using **GridSearchCV**
* Selecting the best performing model
* Exporting the final trained pipeline using **joblib**

---

## 🔍 Model Training & Tuning

* Models are trained inside a unified pipeline
* **GridSearchCV** is used to:

  * Tune hyperparameters
  * Improve model performance
* This ensures a clean and optimized training process

---

## 💾 Model Export

* The complete pipeline (preprocessing + model) is saved using **joblib**
* The exported pipeline can be:

  * Loaded later
  * Used directly for predictions without retraining

---

## 🚀 Production Readiness

This project follows production-ready practices such as:

* Automated preprocessing and training
* Reusable pipeline structure
* Easy model deployment and reuse
* Clean and maintainable code structure

---

## 🧠 Skills Gained

Through this project, the following skills were developed:

* Machine Learning pipeline construction
* Hyperparameter tuning with GridSearchCV
* Model export and reusability
* Building production-ready ML workflows

---

## 📁 Repository Structure (Optional)

```
End-to-End-ML-Pipeline/
│
├── data/
├── notebooks/
├── pipeline.py
├── train.py
├── model.joblib
├── requirements.txt
└── README.md
```

---

## 🧰 Tools & Technologies

* Python
* Scikit-learn
* Pandas
* NumPy
* Joblib

---

## ✅ Conclusion

This project demonstrates how to build a complete and reusable machine learning pipeline using Scikit-learn. It highlights best practices for preprocessing, model training, tuning, and deployment-ready model export.

