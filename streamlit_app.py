import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# Custom styling
st.markdown(
    """
    <style>
        body {
            background-color: #e6f4ea;
            color: #1e1e1e;
        }
        .stApp {
            background-color: #e6f4ea;
        }
        .css-18e3th9, .css-1d391kg {
            background-color: #d8efe0 !important;
        }
        .stSidebar {
            background-color: #c8e6c9;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# University logo and title
st.markdown(
    """
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/8/83/TUMS_Signature_Variation_1_BLUE.png' width='200' style='margin-bottom: 10px;'/>
    </div>
    """,
    unsafe_allow_html=True
)

st.title('🤖🤰 Machine Learning Models APP for Advance Predicting Infertility Risk in Women')
st.info('Predict the **Infertility** based on startup data using Multiple XGBoost.')

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Bahsobi/sii_project/main/cleaned_data%20(3).xlsx"
    return pd.read_excel(url)

df = load_data()

# Rename columns
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

# Select features
features = ['SSI', 'age', 'BMI', 'waist_circumference', 'race', 'hyperlipidemia', 'diabetes']
target = 'infertility'
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

categorical_features = ['race', 'hyperlipidemia', 'diabetes']
numerical_features = ['SSI', 'age', 'BMI', 'waist_circumference']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# XGBoost pipeline
model = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

# Feature importance (XGBoost)
xgb_model = model.named_steps['xgb']
encoder = model.named_steps['prep'].named_transformers_['cat']
feature_names = encoder.get_feature_names_out(categorical_features).tolist() + numerical_features
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Logistic Regression (odds ratio)
odds_pipeline = Pipeline([
    ('prep', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])
odds_pipeline.fit(X_train, y_train)
log_model = odds_pipeline.named_steps['logreg']
log_odds = log_model.coef_[0]
odds_ratios = np.exp(log_odds)

odds_df = pd.DataFrame({
    'Feature': feature_names,
    'Odds Ratio (Infertility)': odds_ratios
}).sort_values(by='Odds Ratio (Infertility)', ascending=False)

# Sidebar for input
temp_input = st.sidebar.container()
temp_input.header("📝 Input Individual Data")

race_options = [
    "Mexican American", "Other Hispanic", "Non-Hispanic White",
    "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race - Including Multi-Racial"
]

ssi = temp_input.number_input("SSI", min_value=0.0, value=10.0)
age = temp_input.number_input("Age", min_value=15, max_value=60, value=30)
bmi = temp_input.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
waist = temp_input.number_input("Waist Circumference", min_value=40.0, max_value=150.0, value=80.0)
race = temp_input.selectbox("Race", race_options)
hyperlipidemia = temp_input.selectbox("Hyperlipidemia", ['Yes', 'No'])
diabetes = temp_input.selectbox("Diabetes", ['Yes', 'No'])

# Prediction
user_input = pd.DataFrame([{
    'SSI': ssi,
    'age': age,
    'BMI': bmi,
    'waist_circumference': waist,
    'race': race,
    'hyperlipidemia': hyperlipidemia,
    'diabetes': diabetes
}])

prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]

# Output result
st.subheader("🔍 Infertility Prediction")
if prediction == 1:
    st.error(f"⚠️ Predicted: *Infertile* with probability {probability:.2%}")
else:
    st.success(f"✅ Predicted: *Not Infertile* with probability {1 - probability:.2%}")

# Odds Ratios (Logistic Regression)
st.subheader("📊 Odds Ratios for Infertility (Logistic Regression)")
st.dataframe(odds_df)

# XGBoost Feature Importances
st.subheader("💡 Feature Importances for Infertility (XGBoost)")
st.dataframe(importance_df)

# Feature Importance Chart
st.subheader("📈 Bar Chart: Feature Importances (XGBoost)")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
st.pyplot(fig)

# Data Summary
with st.expander("📋 Data Summary"):
    st.write(df.describe())

# Infertility Pie Chart
st.subheader("🎯 Infertility Distribution")
fig2, ax2 = plt.subplots()
df['infertility'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Not Infertile', 'Infertile'], ax=ax2)
ax2.set_ylabel("")
st.pyplot(fig2)

# Sample Data
with st.expander("🔍 Sample Data (First 10 Rows)"):
    st.dataframe(df.head(10))
