import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
def load_data(path):
    return pd.read_csv(path)
def clean_data(df):
    df = df.copy()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna("Unknown", inplace=True)
    return df
def feature_engineering(df):
    df["usage_per_month"] = df.iloc[:, 1] / (df.iloc[:, 3] + 1)
    df["complaint_rate"] = df.iloc[:, 2] / (df.iloc[:, 3] + 1)
    df["usage_norm"] = (df.iloc[:, 1] - df.iloc[:, 1].mean()) / df.iloc[:, 1].std()
    return df
def classify_customer(row):
    usage = row.iloc[1]
    complaints = row.iloc[2]
    months_active = row.iloc[3]
    if usage > 80:
        return "High Value"
    if complaints > 3:
        return "At Risk"
    if months_active < 3:
        return "New Customer"
    return "Normal"
def generate_report(df):
    print("\n--- Customer Report ---")
    cols_to_show = df.columns[:4].tolist() + ["class"]
    print(df[cols_to_show].head())
def draw_visuals(df):
    churn_col = df.columns[5]
    country_col = df.columns[4]
    plt.figure()
    sns.countplot(data=df, x=churn_col)
    plt.title("Churn vs Non-Churn")
    plt.show()
    plt.figure()
    df[country_col].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("Customer Countries")
    plt.show()
    plt.figure()
    df.groupby(df.columns[3])[df.columns[1]].mean().plot()
    plt.title("Usage Over Months")
    plt.show()
    plt.figure()
    sns.heatmap(df[[df.columns[1], df.columns[2], df.columns[3]]].corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.show()
def train_models(df):
    X = df.iloc[:, 1:4]
    y = df.iloc[:, 5]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    results = {}
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    results["Logistic Regression"] = (accuracy_score(y_test, lr_preds), f1_score(y_test, lr_preds))
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["Random Forest"] = (accuracy_score(y_test, rf_preds), f1_score(y_test, rf_preds))
    best_model = rf if results["Random Forest"][1] > results["Logistic Regression"][1] else lr
    with open(r"E:\program shii\model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    return results
df = load_data(r"E:\program shii\customers_with_country.csv")
df = clean_data(df)
df = feature_engineering(df)
df["class"] = df.apply(classify_customer, axis=1)
generate_report(df)
draw_visuals(df)
results = train_models(df)
print("\nModel Comparison:", results)

print('--- MADE BY MAZEN ---')