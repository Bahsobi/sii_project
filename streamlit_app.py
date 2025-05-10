import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# --- عنوان ---
st.title("🤰 پیش‌بینی ناباروری زنان با XGBoost")

st.markdown("""
این اپلیکیشن با استفاده از الگوریتم **XGBoost** و ویژگی‌های پزشکی، احتمال ناباروری را پیش‌بینی می‌کند.
""")

# --- بارگذاری داده‌ها ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Bahsobi/sii_project/main/MY_%20ssi.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# --- نمایش داده‌ها ---
st.subheader("🔍 پیش‌نمایش داده‌ها")
st.dataframe(df.head())

# --- آماده‌سازی داده‌ها ---
features = ['SSI', 'age', 'BMI', 'waist circumference', 'race', 'hyperlipidemia', 'diabetes']
target = 'infertility'

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# --- پیش‌پردازش ---
categorical_features = ['race', 'hyperlipidemia', 'diabetes']
numerical_features = ['SSI', 'age', 'BMI', 'waist circumference']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

model = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# --- آموزش مدل ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

# --- فرم ورودی کاربر ---
st.sidebar.header("📝 وارد کردن اطلاعات فردی")

ssi = st.sidebar.number_input("SSI", min_value=0.0, value=10.0)
age = st.sidebar.number_input("Age", min_value=15, max_value=60, value=30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
waist = st.sidebar.number_input("Waist Circumference", min_value=40.0, max_value=150.0, value=80.0)

race_options = sorted(df['race'].dropna().unique())
race = st.sidebar.selectbox("Race", race_options)

hyperlipidemia = st.sidebar.selectbox("Hyperlipidemia", ['Yes', 'No'])
diabetes = st.sidebar.selectbox("Diabetes", ['Yes', 'No'])

# --- آماده‌سازی ورودی ---
input_data = pd.DataFrame([{
    'SSI': ssi,
    'age': age,
    'BMI': bmi,
    'waist circumference': waist,
    'race': race,
    'hyperlipidemia': hyperlipidemia,
    'diabetes': diabetes
}])

# --- پیش‌بینی ---
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# --- نمایش نتیجه ---
st.subheader("🔎 نتیجه پیش‌بینی")

if prediction == 1:
    st.error(f"⚠️ احتمال ناباروری: {probability:.1%}")
else:
    st.success(f"✅ احتمال عدم ناباروری: {(1 - probability):.1%}")
