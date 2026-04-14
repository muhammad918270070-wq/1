import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIG ---
# Настройка заголовка вкладки браузера и широкого формата страницы
st.set_page_config(page_title="🏠 House Price AI", layout="wide")

# Заголовки на главной странице
st.title("🏠 AI House Price Predictor")
st.markdown("### Smart real estate valuation powered by Machine Learning")

# --- LOAD DATA ---
# Кэшируем данные, чтобы они не перегружались при каждом взаимодействии с интерфейсом
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    # Создаем DataFrame с признаками (X) и целевую переменную с ценой (y)
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

X, y = load_data()

# --- MODEL ---
# Кэшируем модель, чтобы не переобучать её при каждом изменении слайдера
@st.cache_resource
def train_model():
    # Создаем модель случайного леса со 150 деревьями
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y) # Обучаем на загруженных данных
    return model

model = train_model()

# --- SIDEBAR ---
# Заголовок боковой панели
st.sidebar.title("⚙️ Customize House")

# Функция для создания слайдеров на основе признаков из датасета
def user_input():
    data = {}
    for col in X.columns:
        # Создаем слайдер для каждого признака с мин, макс и средним значением
        data[col] = st.sidebar.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
    # Возвращаем введенные данные в виде одной строки таблицы (DataFrame)
    return pd.DataFrame(data, index=[0])

df = user_input() # Получаем данные от пользователя

# Расчет предсказания
prediction = model.predict(df)[0]
# Масштабируем цену (в датасете значения в сотнях тысяч долларов)
price = prediction * 100000 

# --- TOP METRICS ---
# Создаем три колонки для вывода ключевых показателей
col1, col2, col3 = st.columns(3)

# Выводим предсказанную цену, среднюю по датасету и разницу между ними
col1.metric("💰 Predicted Price", f"${price:,.0f}")
col2.metric("📊 Dataset Avg", f"${y.mean()*100000:,.0f}")
col3.metric("📝 Difference", f"${(price - y.mean()*100000):,.0f}")

st.divider() # Разделительная линия

# --- TABS ---
# Создаем вкладки с названиями в точности как на твоем макете
tab1, tab2, tab3, tab4 = st.tabs(["📥 Input", "📊 Analytics", "🗺️ Map", "🧠 Model"])

with tab1:
    st.subheader("Your House Parameters")
    # Показываем таблицу с данными, которые пользователь выбрал слайдерами
    st.dataframe(df)

with tab2:
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Price Distribution")
        # Гистограмма распределения цен в датасете
        fig, ax = plt.subplots()
        ax.hist(y, bins=30)
        # Добавляем вертикальную линию нашего предсказания (в том же масштабе, что и y)
        ax.axvline(prediction, color='red', linestyle='--')
        st.pyplot(fig)

    with colB:
        st.subheader("Feature Importance")
        # Создаем таблицу важности признаков для графика
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
        
        # Строим горизонтальный столбчатый график
        fig2, ax2 = plt.subplots()
        ax2.barh(importance["Feature"], importance["Importance"])
        st.pyplot(fig2)

with tab3:
    st.subheader("Geographic Visualization")
    # Готовим данные для карты, переименовывая широту и долготу под стандарт st.map
    map_data = X.copy()
    map_data['price'] = y
    st.map(map_data.rename(columns={"Latitude": "lat", "Longitude": "lon"}))
with tab4:
    st.subheader("Model Explanation")
    # Текстовое описание используемой модели
    st.write("""
    This model uses Random Forest Regression.
    
    ✓ Combines multiple decision trees
    ✓ Captures complex relationships
    ✓ Works well for real estate data
    """)
    
    st.subheader("Correlation Matrix")
    # Вычисляем корреляцию признаков и строим матрицу
    corr = X.corr()
    fig3, ax3 = plt.subplots()
    cax = ax3.matshow(corr)
    # Настраиваем оси и цветовую шкалу
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig3.colorbar(cax)
    st.pyplot(fig3)

# --- BONUS ---
st.divider()
# Сравнение параметров пользователя со средними значениями по датасету
st.subheader("📊 Compare Your House to Dataset")
comparison = pd.DataFrame({
    "Your House": df.iloc[0],
    "Average": X.mean()
})
st.bar_chart(comparison)
