"""
train_model.py
Bank Customer Churn Prediction — end-to-end training script.
Reads Churn_Modelling.csv, cleans/encodes/engineers features,
trains a logistic regression churn model, and exports:
  - churn_scored_data.csv (scored dataset for Power BI)
  - feature_importance.csv (model coefficients for Power BI)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------- 1. Load ----------
df = pd.read_csv('Churn_Modelling.csv')

# ---------- 2. Clean ----------
df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# ---------- 3. Encode ----------
df_clean['Gender'] = df_clean['Gender'].map({'Female': 0, 'Male': 1})
df_clean = pd.get_dummies(df_clean, columns=['Geography'], drop_first=True)

# ---------- 4. Feature engineering ----------
def age_group(age):
    if age <= 30:
        return 'Young'
    elif age <= 45:
        return 'Middle'
    elif age <= 60:
        return 'Senior'
    else:
        return 'Elder'

df_clean['AgeGroup'] = df_clean['Age'].apply(age_group)

# CLV estimate — see "How CLV is calculated" in README.
# Not a bank-reported figure: a custom proxy combining current
# balance with a tenure-weighted contribution from income.
df_clean['CLV'] = df_clean['Balance'] + (df_clean['EstimatedSalary'] * df_clean['Tenure'] * 0.1)

# ---------- 5. Prepare features/target ----------
X = df_clean.drop(['Exited', 'AgeGroup'], axis=1)
y = df_clean['Exited']

# ---------- 6. Scale ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 7. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------- 8. Train model ----------
# class_weight='balanced' addresses class imbalance (~20% churn rate).
# Without it, recall on churners was 0.20 (81% accuracy, but missed
# 4 out of 5 real churners). With it: 0.71 recall, 0.72 accuracy —
# a deliberate trade-off, since a missed churner costs a customer
# permanently while a false alarm costs a retention offer.
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# ---------- 9. Evaluate ----------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------- 10. Score all customers ----------
df_clean['ChurnProbability'] = model.predict_proba(X_scaled)[:, 1]
df_clean['ChurnRiskFlag'] = (df_clean['ChurnProbability'] > 0.5).astype(int)
df_clean['CustomerId'] = df['CustomerId']
df_clean.to_csv('churn_scored_data.csv', index=False)

# ---------- 11. Feature importance ----------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
feature_importance['AbsCoefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values('AbsCoefficient', ascending=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print("\nDone. Exported churn_scored_data.csv and feature_importance.csv")