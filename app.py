import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ZARA Sales Predictor", layout="wide")
st.title("🛍️ Zara Sales Volume Prediction")

# Load dataset
df = pd.read_csv("zara.csv", delimiter=';')
df.drop_duplicates(inplace=True)
df['name'].fillna("Unknown", inplace=True)
df['description'].fillna("No description", inplace=True)
df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['Sales Volume'] = pd.to_numeric(df['Sales Volume'], errors='coerce')
df.dropna(subset=['price', 'Sales Volume'], inplace=True)

st.subheader("📊 Dataset Preview")
st.dataframe(df[['name', 'price', 'Sales Volume', 'Product Position', 'Promotion']].head())

# EDA Chart
st.subheader("📈 Price vs Sales Volume")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='price', y='Sales Volume', ax=ax)
st.pyplot(fig)

# Model Training
X = df[['price']]
y = df['Sales Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📉 Model Evaluation")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R²):** {r2:.4f}")

# Prediction
st.subheader("🔮 Predict Sales Volume")
input_price = st.number_input("Enter product price (USD):", min_value=0.0, step=1.0)
if st.button("Predict"):
    prediction = model.predict([[input_price]])
    st.success(f"Estimated Sales Volume: {int(prediction[0])}")
