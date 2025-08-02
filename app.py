import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

st.set_page_config(page_title="📈 Dự đoán Sales_After", layout="wide")
st.title("💼 Ứng dụng học máy: Dự đoán doanh số sau chiến dịch")

uploaded_file = st.file_uploader("📥 Tải lên file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dữ liệu gốc (5 dòng đầu)")
    st.dataframe(df.head())

    # ======= 1. TIỀN XỬ LÝ =======
    df = df.dropna()
    df = df.drop_duplicates()

    le_group = LabelEncoder()
    le_segment = LabelEncoder()

    df['Group'] = le_group.fit_transform(df['Group'])
    df['Customer_Segment'] = le_segment.fit_transform(df['Customer_Segment'])

    # Lưu ý: map lại nếu dùng cho dự đoán sau này
    if 'Purchase_Made' in df.columns:
        df['Purchase_Made'] = df['Purchase_Made'].map({'Yes': 1, 'No': 0})

    scaler = MinMaxScaler()
    cols_to_scale = ['Sales_Before', 'Customer_Satisfaction_Before', 'Customer_Satisfaction_After']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    st.success("✅ Đã xử lý dữ liệu: xóa null, duplicate, chuẩn hóa")

    # ======= 2. PHÂN TÍCH DỮ LIỆU =======
    st.subheader("📊 Thống kê mô tả")
    st.dataframe(df.describe())

    st.subheader("📈 Ma trận tương quan")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # ======= 3. TRỰC QUAN HÓA =======
    st.subheader("🧩 Trực quan hóa")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pie Chart: Customer Segment**")
        fig1, ax1 = plt.subplots()
        df['Customer_Segment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col2:
        st.markdown("**Pie Chart: Purchase Made**")
        fig2, ax2 = plt.subplots()
        df['Purchase_Made'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Bar Chart: Sales_After theo Group**")
        fig3, ax3 = plt.subplots()
        df.groupby('Group')['Sales_After'].mean().plot(kind='bar', ax=ax3)
        st.pyplot(fig3)

    with col4:
        st.markdown("**Bar Chart: Satisfaction_After theo Segment**")
        fig4, ax4 = plt.subplots()
        df.groupby('Customer_Segment')['Customer_Satisfaction_After'].mean().plot(kind='bar', ax=ax4)
        st.pyplot(fig4)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Line Chart: Sales Before vs After theo Group**")
        fig5, ax5 = plt.subplots()
        df.groupby('Group')[['Sales_Before', 'Sales_After']].mean().plot(ax=ax5, marker='o')
        st.pyplot(fig5)

    with col6:
        st.markdown("**Line Chart: Satisfaction Before vs After theo Segment**")
        fig6, ax6 = plt.subplots()
        df.groupby('Customer_Segment')[['Customer_Satisfaction_Before', 'Customer_Satisfaction_After']].mean().plot(ax=ax6, marker='s')
        st.pyplot(fig6)

    # ======= 4. HUẤN LUYỆN MÔ HÌNH =======
    st.subheader("🧠 Huấn luyện mô hình dự đoán Sales_After")

    X = df.drop(columns=['Sales_After', 'Purchase_Made'], errors='ignore')
    y = df['Sales_After']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"📉 MSE: **{mse:.2f}**")
    st.markdown(f"📈 R² Score: **{r2:.4f}**")

    # ======= 5. DỰ ĐOÁN TRỰC TIẾP =======
    st.subheader("🎯 Dự đoán Sales_After từ dữ liệu hiện tại")

    df['Dự đoán Sales_After'] = model.predict(X)
    st.dataframe(df[['Sales_After', 'Dự đoán Sales_After']].head())

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Tải kết quả về CSV", data=csv, file_name="du_doan_sales_after.csv", mime="text/csv")
