import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIG ---
st.set_page_config(page_title="🏠 House Price AI", layout="wide")

st.title("🏠 AI House Price Predictor")
st.markdown("### Smart real estate valuation powered by Machine Learning")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

X, y = load_data()

# --- MODEL TRAINING ---
@st.cache_resource
def train_model():
    # Используем 150 деревьев для баланса скорости и точности
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# --- SIDEBAR ---
st.sidebar.title("⚙️ Customize House")
st.sidebar.divider()

def user_input():
    data = {}
    for col in X.columns:
        # Автоматическая генерация слайдеров на основе данных
        data[col] = st.sidebar.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
    return pd.DataFrame(data, index=[0])

df_input = user_input()

# --- PREDICTION LOGIC ---
prediction = model.predict(df_input)[0]
price = prediction * 100000  # Перевод в доллары
avg_price = y.mean() * 100000
diff = price - avg_price

# --- TOP METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("💰 Predicted Price", f"${price:,.0f}")
col2.metric("📊 Dataset Avg", f"${avg_price:,.0f}")
col3.metric("📈 Difference", f"${diff:,.0f}", delta_color="normal")

st.divider()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📋 Input", "📊 Analytics", "🗺️ Map", "🧠 Model Info"])

with tab1:
    st.subheader("Your House Parameters")
    st.dataframe(df_input)

with tab2:
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(y * 100000, bins=30, kde=True, ax=ax)
        ax.axvline(price, color='red', linestyle='--', label='Your Prediction')
        plt.legend()
        st.pyplot(fig)

    with colB:
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
        
        fig2, ax2 = plt.subplots()
        ax2.barh(importance["Feature"], importance["Importance"], color='skyblue')
        st.pyplot(fig2)

with tab3:
    st.subheader("Geographic Visualization")
    # Подготовка данных для карты (Streamlit нужны колонки 'lat' и 'lon')
    map_data = X.copy()
    map_data['price'] = y
    map_data = map_data.rename(columns={"Latitude": "lat", "Longitude": "lon"})
    st.map(map_data)

with tab4:
    st.subheader("Model Explanation")
    st.write("""
    This model uses Random Forest Regression to estimate house prices in California.
    - ✅ Combines 150 decision trees for robust predictions.
    - ✅ Captures complex relationships between location and income.
    - ✅ Handles non-linear data better than simple linear regression.
    """)
    
    st.subheader("Correlation Matrix")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
    st.pyplot(fig3)

# --- BONUS: COMPARISON ---
st.divider()
st.subheader("📊 Compare Your House to Dataset Average")
comparison = pd.DataFrame({
    "Your House": df_input.iloc[0],
    "Average": X.mean()
})
st.bar_chart(comparison)
