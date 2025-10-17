AI for Quality Health: Heart Disease Prediction

Course: Machine Learning / Artificial Intelligence

INTRODUCTION 

This project focuses on UN Sustainable Development Goal (SDG) 3: Good Health and Well-being, specifically targeting the early detection and prevention of heart disease through Artificial Intelligence. Cardiovascular diseases remain one of the leading causes of death globally, and early prediction can significantly improve patient outcomes.

Using supervised machine learning, this project develops a predictive model that identifies individuals at high risk of heart disease based on clinical and lifestyle data.

PROBLEM STATEMENT

Heart disease prediction using traditional methods often relies heavily on manual interpretation and delayed diagnosis. 
The challenge is to design an AI-driven solution capable of predicting heart disease risk accurately and efficiently from structured patient data, supporting medical professionals in making informed decisions.

OBJECTIVE

To build and evaluate a supervised machine learning model that predicts whether an individual has heart disease based on health parameters such as age, cholesterol level, blood pressure, and other clinical indicators.

DATASET

The dataset used in this study is the Heart Disease dataset from the UCI Machine Learning Repository, which contains 303 patient records with the following attributes:

Numerical features: Age, Resting Blood Pressure, Serum Cholesterol, Maximum Heart Rate, Oldpeak, etc.

Categorical features: Sex, Chest Pain Type, Fasting Blood Sugar, Exercise-Induced Angina, etc.

Target variable: Presence of heart disease (1 = Disease, 0 = No Disease)

5. METHODOLODY
 - Data Preprocessing

Missing Value Handling: Numerical data imputed using median values; categorical data filled with “missing” category.

Feature Scaling: StandardScaler applied to numerical columns.

Encoding: OneHotEncoder used to convert categorical variables into numerical form.

 -Model Design

The model uses a Logistic Regression classifier within a preprocessing pipeline.
The pipeline integrates:

-Data cleaning and transformation steps

Feature encoding and scaling

Model training and prediction

- Training and Evaluation

The dataset is split into 80% training and 20% testing sets.

The model performance is measured using Accuracy, Precision, Recall, F1-score, and AUC-ROC.

6. RESULTS

The Logistic Regression model achieved:

Accuracy: ~85%

Precision: 0.83

Recall: 0.86

AUC Score: 0.91

These metrics indicate strong performance in distinguishing between patients with and without heart disease.

7. Discussion

The model demonstrates that machine learning can provide reliable predictions using patient health indicators. Such systems can support medical practitioners in early diagnosis and screening programs, contributing to SDG 3 by improving health outcomes and reducing preventable deaths.

However, the model’s accuracy can be further improved by:

Using ensemble methods like Random Forest or XGBoost

Expanding the dataset to include diverse populations

Integrating real-time health monitoring data (e.g., from wearable devices)

8. Conclusion

This project successfully demonstrates how supervised learning can be applied to real-world healthcare challenges. The trained model predicts heart disease risk effectively, aligning with the UN SDG goal of ensuring healthy lives and promoting well-being for all at all ages.

9. References

UCI Machine Learning Repository: Heart Disease Dataset

Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, 2011.

United Nations, “Sustainable Development Goal 3: Good Health and Well-being,” 2015.

10. Appendix
A. Tools and Libraries

Python 3.12

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn 

Google Colab

Final Deliverable:
A trained AI model capable of predicting heart disease risk using clinical data. This aligns AI innovation with the pursuit of global health equity, showcasing technology’s potential to advance sustainable development and human well-being.
