import streamlit as st
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title('My first project')
st.markdown("### Smart real estate valuation powered by Machine Learning")

@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

X, y = load_data()

# ---------------------------
# MODEL
# ---------------------------
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=150)
    model.fit(X, y)
    return model

model = train_model()
