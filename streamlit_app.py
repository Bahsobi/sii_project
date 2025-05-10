import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# عنوان
st.title("🤰 پیش‌بینی ناباروری با XGBoost")

# بارگذاری داده‌ها
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Bahsobi/sii_project/main/cleaned_data%20(3).xlsx"
    return pd.read_excel(url)

df = load_data()

# نمایش ستون‌ها برای رفع خطا
st.subheader("ستون‌های موجود در داده‌ها")
st.write(df.columns.tolist())

# تغییر نام ستون‌ها برای یکدستی
df.rename(columns={
    'AGE': 'age',
    'Race': 'race',
    'BMI': 'BMI',
    'Waist Circumference': 'waist_circumference',
    'Hyperlipidemia': 'hyperlipidemia',
    'diabetes': 'diabetes',
    'SSI': 'SSI',
    'Female infertility': 'infertility'
}, inplace=True)

# تعریف ویژگی‌ها و هدف
features = ['SSI', 'age', 'BMI', 'waist_circumference', 'race', 'hyperlipidemia', 'diabetes']
target = 'infertility'

# فیلتر کردن فقط ستون‌های موردنیاز و حذف مقادیر گمشده
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# پیش‌پردازش
categorical_features = ['race', 'hyperlipidemia', 'diabetes']
numerical_features = ['SSI', 'age', 'BMI', 'waist_circumference']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# ساخت مدل
model = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# آموزش مدل
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

# فرم ورودی کاربر
st.sidebar.header("📝 وارد کردن اطلاعات فردی")

ssi = st.sidebar.number_input("SSI", min_value=0.0, value=10.0)
age = st.sidebar.number_input("Age", min_value=15, max_value=60, value=30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
waist = st.sidebar.number_input("Waist Circumference", min_value=40.0, max_value=150.0, value=80.0)

race_options = [
    "Mexican American", "Other Hispanic", "Non-Hispanic White",
    "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race - Including Multi-Racial"
]
race = st.sidebar.selectbox("Race", race_options)
hyperlipidemia = st.sidebar.selectbox("Hyperlipidemia", ['Yes', 'No'])
diabetes = st.sidebar.selectbox("Diabetes", ['Yes', 'No'])

# آماده‌سازی ورودی کاربر
user_input = pd.DataFrame([{
    'SSI': ssi,
    'age': age,
    'BMI': bmi,
    'waist_circumference': waist,
    'race': race,
    'hyperlipidemia': hyperlipidemia,
    'diabetes': diabetes
}])

# پیش‌بینی
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]

# نمایش نتیجه
st.subheader("🔍 پیش‌بینی ناباروری")
if prediction == 1:
    st.error(f"⚠️ پیش‌بینی شده: *ناباروری* با احتمال {probability:.2%}")
else:
    st.success(f"✅ پیش‌بینی شده: *عدم ناباروری* با احتمال {1 - probability:.2%}")
