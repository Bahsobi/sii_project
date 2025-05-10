import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# عنوان اپ
st.title("🤰 پیش‌بینی ناباروری با XGBoost")

# نگاشت کد نژاد به برچسب
race_map = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    5: "Non-Hispanic Asian",
    6: "Other Race - Including Multi-Racial"
}
race_map_inv = {v: k for k, v in race_map.items()}

# بارگذاری داده‌ها
@st.cache_data
def load_data():
    return pd.read_excel("/mnt/data/cleaned_data (3).xlsx")

df = load_data()

# جایگزینی کدهای نژاد با لیبل برای نمایش بهتر
df['Race'] = df['Race'].map(race_map)

st.subheader("📊 نمایش داده‌های اولیه")
st.dataframe(df.head())

# انتخاب ویژگی‌ها و هدف
features = ['SSI', 'AGE', 'BMI', 'Waist Circumference', 'Race', 'Hyperlipidemia', 'diabetes']
target = 'Infertility'

# حذف مقادیر گمشده
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# پیش‌پردازش داده‌ها
categorical_features = ['Race', 'Hyperlipidemia', 'diabetes']
numerical_features = ['SSI', 'AGE', 'BMI', 'Waist Circumference']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# مدل نهایی با XGBoost
model = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# آموزش مدل
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# فرم ورودی کاربر
st.sidebar.header("📝 وارد کردن اطلاعات فردی")

ssi = st.sidebar.number_input("SSI", min_value=0.0, value=10.0)
age = st.sidebar.number_input("Age", min_value=15, max_value=60, value=30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
waist = st.sidebar.number_input("Waist Circumference", min_value=40.0, max_value=150.0, value=80.0)

race_label = st.sidebar.selectbox("Race", list(race_map.values()))
race = race_label

hyperlipidemia = st.sidebar.selectbox("Hyperlipidemia", ['Yes', 'No'])
diabetes = st.sidebar.selectbox("Diabetes", ['Yes', 'No'])

# ساخت دیتافریم ورودی
user_input = pd.DataFrame([{
    'SSI': ssi,
    'AGE': age,
    'BMI': bmi,
    'Waist Circumference': waist,
    'Race': race,
    'Hyperlipidemia': hyperlipidemia,
    'diabetes': diabetes
}])

# پیش‌بینی
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]

# نمایش خروجی
st.subheader("🔍 پیش‌بینی ناباروری")
if prediction == 1:
    st.error(f"⚠️ پیش‌بینی شده: *ناباروری* با احتمال {probability:.2%}")
else:
    st.success(f"✅ پیش‌بینی شده: *عدم ناباروری* با احتمال {1 - probability:.2%}")
