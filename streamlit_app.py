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
import statsmodels.api as sm

# ---------- Custom Styling ----------
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

# ---------- Header ----------
st.markdown(
    """
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/8/83/TUMS_Signature_Variation_1_BLUE.png' width='200' style='margin-bottom: 10px;'/>
    </div>
    """,
    unsafe_allow_html=True
)

st.title('🤖🤰 Machine Learning Models APP for Advance Predicting Infertility Risk in Women')
st.info('Predict the **Infertility** based on health data using XGBoost and Logistic Regression.')

# ---------- Load Data ----------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Bahsobi/sii_project/main/cleaned_data%20(3).xlsx"
    return pd.read_excel(url)

df = load_data()

# ---------- Rename Columns ----------
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

# ---------- Features & Target ----------
features = ['SSI', 'age', 'BMI', 'waist_circumference', 'race', 'hyperlipidemia', 'diabetes']
target = 'infertility'
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# ---------- Preprocessing ----------
categorical_features = ['race', 'hyperlipidemia', 'diabetes']
numerical_features = ['SSI', 'age', 'BMI', 'waist_circumference']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# ---------- XGBoost Pipeline ----------
model = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

# ---------- Feature Importance from XGBoost ----------
xgb_model = model.named_steps['xgb']
encoder = model.named_steps['prep'].named_transformers_['cat']
feature_names = encoder.get_feature_names_out(categorical_features).tolist() + numerical_features
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# ---------- Logistic Regression for Odds Ratio ----------
odds_pipeline = Pipeline([
    ('prep', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])
odds_pipeline.fit(X_train, y_train)
log_model = odds_pipeline.named_steps['logreg']
odds_ratios = np.exp(log_model.coef_[0])

odds_df = pd.DataFrame({
    'Feature': feature_names,
    'Odds Ratio': odds_ratios
}).sort_values(by='Odds Ratio', ascending=False)

# 🔴 حذف ویژگی‌های نژادی از جدول Odds Ratios
filtered_odds_df = odds_df[~odds_df['Feature'].str.contains("race")]

# ---------- Sidebar User Input ----------
st.sidebar.header("📝 Input Individual Data")
race_options = [
    "Mexican American", "Other Hispanic", "Non-Hispanic White",
    "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race - Including Multi-Racial"
]

ssi = st.sidebar.number_input("SSI", min_value=0.0, value=10.0)
age = st.sidebar.number_input("Age", min_value=15, max_value=60, value=30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
waist = st.sidebar.number_input("Waist Circumference", min_value=40.0, max_value=150.0, value=80.0)
race = st.sidebar.selectbox("Race", race_options)
hyperlipidemia = st.sidebar.selectbox("Hyperlipidemia", ['Yes', 'No'])
diabetes = st.sidebar.selectbox("Diabetes", ['Yes', 'No'])

# ---------- Prediction ----------
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
odds_value = probability / (1 - probability)

# ---------- Display Result ----------
if prediction == 1:
    st.error(f"""
        ⚠️ **Prediction: Infertile**

        🧮 **Probability of Infertility:** {probability:.2%}  
        🎲 **Odds of Infertility:** {odds_value:.2f}
    """)
else:
    st.success(f"""
        ✅ **Prediction: Not Infertile**

        🧮 **Probability of Infertility:** {probability:.2%}  
        🎲 **Odds of Infertility:** {odds_value:.2f}
    """)



# ---------- Show Odds Ratios Table ----------
st.subheader("📊 Odds Ratios for Infertility (Logistic Regression) (Excluding Race)")
st.dataframe(filtered_odds_df)

# ---------- Show XGBoost Feature Importance Table ----------
st.subheader("💡 Feature Importances (XGBoost)")
st.dataframe(importance_df)

# ---------- Plot XGBoost Feature Importances ----------
st.subheader("📈 Bar Chart: Feature Importances")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
st.pyplot(fig)

# ---------- Odds Ratio for SII Quartiles ----------
st.subheader("📉 Odds Ratios for Infertility by SII Quartiles")
df_sii = df[['SSI', 'infertility']].copy()
df_sii['SII_quartile'] = pd.qcut(df_sii['SSI'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# تبدیل به داده عددی برای مدل logit
X_q = pd.get_dummies(df_sii['SII_quartile'], drop_first=True)
X_q = sm.add_constant(X_q).astype(float)
y_q = df_sii['infertility'].astype(float)

model_q = sm.Logit(y_q, X_q).fit(disp=False)
ors = np.exp(model_q.params)
ci = model_q.conf_int()
ci.columns = ['2.5%', '97.5%']
ci = np.exp(ci)

or_df = pd.DataFrame({
    'Quartile': ors.index,
    'Odds Ratio': ors.values,
    'CI Lower': ci['2.5%'],
    'CI Upper': ci['97.5%'],
    'p-value': model_q.pvalues
}).query("Quartile != 'const'")

st.dataframe(or_df.set_index('Quartile').style.format("{:.2f}"))

# ---------- Plot Quartile OR ----------
fig3, ax3 = plt.subplots()
sns.pointplot(data=or_df, x='Quartile', y='Odds Ratio', join=False, capsize=0.2, errwidth=1.5)
ax3.axhline(1, linestyle='--', color='gray')
ax3.set_title("Odds Ratios for Infertility by SII Quartiles")
st.pyplot(fig3)

# ---------- Data Summary ----------
with st.expander("📋 Data Summary"):
    st.write(df.describe())

# ---------- Pie Chart: Infertility Distribution ----------
st.subheader("🎯 Infertility Distribution")
fig2, ax2 = plt.subplots()
df['infertility'].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['Not Infertile', 'Infertile'], ax=ax2, colors=["#81c784", "#e57373"])
ax2.set_ylabel("")
st.pyplot(fig2)

# ---------- Sample Data ----------
with st.expander("🔍 Sample Data (First 10 Rows)"):
    st.dataframe(df.head(10))
