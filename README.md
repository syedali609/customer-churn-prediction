# 📉 Customer Churn Prediction

A machine learning project to predict whether a telecom customer will churn or stay.

---

## 📊 Dataset
- **Source:** IBM Telco Customer Churn — Kaggle
- **Rows:** 7,043 customers | **Columns:** 21
- **Churn Rate:** ~26%

---

## ⚙️ Project Steps
1. **EDA** — Explored churn patterns across contract, tenure, and billing
2. **Preprocessing** — Cleaned, encoded, scaled and split the data
3. **Modeling** — Trained and compared 3 ML models
4. **Evaluation** — Confusion Matrix, ROC Curve, Feature Importance
5. **Deployment** — Built a Streamlit web app for real time prediction

---

## 🏆 Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 80.06% | 0.59 | 0.84 |
| Random Forest | 79.13% | 0.55 | 0.82 |
| XGBoost | 77.86% | 0.55 | 0.81 |

✅ Logistic Regression selected as final model. the dataset was not too large because of it 
Random Forest and XGBoost doesn't perform well.

---

## 🔑 Key Findings
- Contract type is the #1 driver of churn
- Fiber optic users churn the most
- Customers with no Tech Support churn more
- Shorter tenure = higher churn risk

---

## 🛠️ Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit

---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app/frontend/app.py
```

---

## Author
**Syed Ali Hussain** — BBA-BIA Student | Aspiring Data Engineer & Analyst 
[LinkedIn](https://www.linkedin.com/in/syed-ali-68482729a)
