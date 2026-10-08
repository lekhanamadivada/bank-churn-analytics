# 📊 Bank Customer Churn & Revenue Risk Analytics

Personal project analyzing customer churn using a public Kaggle dataset (10,000 bank customers) — built end-to-end with Python and Power BI.

## 🔍 What this project does
- 🧹 Cleans and engineers features from raw customer data (Python, Pandas)
- 🤖 Trains a logistic regression model to predict churn (scikit-learn)
- ⚖️ Addresses class imbalance to improve detection of at-risk customers
- 📈 Visualizes findings in a 4-page interactive Power BI dashboard

## 🖥️ Dashboard preview

![Customer Overview](Customer%20Overview.png)
![Churn Drivers](Churn%20Drivers.png)
![Revenue at Risk](Revenue%20at%20Risk.png)
![Model Insights](Model%20Insights.png)

## 💡 Key findings
- 👴 Age is the strongest churn predictor — nearly 2x more influential than any other factor
- 🔁 Customers holding 3–4 products churn at 80–100%, the opposite of the "more products = more loyal" assumption
- 💰 547.79M in customer lifetime value sits with high-risk customers alone

## 📊 Model performance

| Version | Accuracy | Recall (churners) | Precision (churners) |
|---|---|---|---|
| Baseline logistic regression | 81.25% | 0.20 | 0.56 |
| + Feature scaling | 80.9% | 0.20 | 0.54 |
| + `class_weight='balanced'` (final) | 71.9% | **0.71** | 0.38 |

The dataset is imbalanced (~20% churn rate). A baseline model scored 81% accuracy but only caught 1 in 5 actual churners — it had learned to mostly predict "stayed." Using `class_weight='balanced'` in scikit-learn's `LogisticRegression` traded some accuracy and precision for a large recall gain (20% → 71%), a deliberate choice: missing a real churner costs a customer permanently, while a false alarm costs a retention offer.

## 💰 How CLV is calculated

The "customer lifetime value" figure used throughout this project (e.g. 547.79M at risk) is **not a bank-reported financial metric** — it's a custom-engineered proxy built for this project, calculated as:


The logic: current balance is the customer's known value today; the salary/tenure term adds a rough proxy for their earning relationship with the bank over time, weighted down (×0.1) so it doesn't dominate the balance term. This is a simplification for portfolio purposes, not a production-grade CLV model.

## ▶️ How to run this

1. Download the dataset: [Churn for Bank Customers (Kaggle)](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)
2. Place `Churn_Modelling.csv` in this folder
3. `pip install -r requirements.txt`
4. `python train_model.py`
5. Open the `.pbix` file in Power BI Desktop, pointing it at the newly generated `churn_scored_data.csv` and `feature_importance.csv`

## 🛠️ Tools
Python (Pandas, scikit-learn) · Power BI (DAX, Power Query) · Kaggle

## 📁 Files
- `train_model.py` — data cleaning, feature engineering, model training, scoring, and export (fully reproducible)
- `churn_scored_data.csv` — final scored dataset
- `feature_importance.csv` — model coefficients used in the Model Insights dashboard page
- `requirements.txt` — Python dependencies
- `Bank Customer Churn & Revenue Risk Analytics.pbix` — Power BI file
