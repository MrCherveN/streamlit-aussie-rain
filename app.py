import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# завантаження параметрів моделі
model_path = os.path.join(BASE_DIR, "models", "aussie_rain.joblib")

@st.cache_resource
def model_load():
    return joblib.load(model_path)

param = model_load()

model = param["model"]
imputer = param["imputer"]
scaler = param["scaler"]
encoder = param["encoder"]
numeric_cols = param["numeric_cols"]
categorical_cols = param["categorical_cols"]
encoded_cols = param["encoded_cols"]

# завантаження датафрейму
df_path = os.path.join(BASE_DIR, "data", "weatherAUS.csv.zip")
df = pd.read_csv(df_path)

# завантаження зображення
image_path = os.path.join(BASE_DIR, "images", "KWfcpcO.png")

#збір статистики полів для інтерфейсу
stats = df[numeric_cols].agg(['min', 'max', 'mean']).T

#візьмемо діапазони з запасом на 25% від їх значень
stats['min'] = stats['min'] * 0.75
stats['max'] = stats['max'] * 1.25

# Заголовок застосунку
st.title('Прогнозування погоди в Австралії')
st.markdown('Модель використовується для прогнозування ймовірності того, що завтра піде дощ, на основі введених користувачем даних про погоду.')
st.image(image_path)

st.subheader("Вхідні параметри для прогнозу")
st.subheader("Погодні параметри")
data_to_predict = {}

for col in numeric_cols:
    data_to_predict[col] = st.slider(
        col,
        min_value = float(stats.loc[col, "min"]),
        max_value = float(stats.loc[col, "max"]),
        value = float(stats.loc[col, "mean"])
    )

st.subheader("Місцезнаходження та параметри вітру")

cat_unique = {
    col: sorted(df[col].dropna().unique())
    for col in categorical_cols
}

for col in categorical_cols:
    data_to_predict[col] = st.selectbox(col, cat_unique[col])

#обробка введених даних
def preprocess(data):

    data[numeric_cols] = imputer.transform(data[numeric_cols])
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    encoded = pd.DataFrame(encoder.transform(data[categorical_cols]), columns=encoded_cols)

    X = data.drop(columns=categorical_cols)
    X = pd.concat([X, encoded], axis=1)

    return X
# прогноз моделі
def predict(data_to_predict):
    data_to_predict = pd.DataFrame([data_to_predict])
    X = preprocess(data_to_predict)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return pred, proba
    
if st.button("Прогнозувати погоду"):
    
    pred, proba = predict(data_to_predict)
    
    st.write(f"Прогноз погоди на завтра:")
    
    if pred == "Yes":
        st.write("Завтра буде дощ.")
    else:
        st.write("Завтра без дощу.")

    st.write(f"Ймовірність дощу: {proba:.2%}")
