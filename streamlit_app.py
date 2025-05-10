import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier



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









# Show University of Tehran logo and app title centered at the top
st.markdown(
    """
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/8/83/TUMS_Signature_Variation_1_BLUE.png' width='200' style='margin-bottom: 10px;'/>
    </div>
    """,
    unsafe_allow_html=True
)


# App title and description
st.title('🤖 Machine Learning Models APP for Advance Predicting Infertility Risk in Women using XGBoost')
st.info('Predict the **Profit** based on startup data using Multiple Linear Regression.')




# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Bahsobi/sii_project/main/cleaned_data%20(3).xlsx"
    return pd.read_excel(url)

df = load_data()

# Display column names for debugging
st.subheader("Available Columns in the Dataset")
st.write(df.columns.tolist())

# Rename columns for consistency
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

# Feature and target selection
features = ['SSI', 'age', 'BMI', 'waist_circumference', 'race', 'hyperlipidemia', 'diabetes']
target = 'infertility'

# Filter and clean data
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Preprocessing
categorical_features = ['race', 'hyperlipidemia', 'diabetes']
numerical_features = ['SSI', 'age', 'BMI', 'waist_circumference']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Define model pipeline
model = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("📝 Enter Personal Information")

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

# Create input DataFrame
user_input = pd.DataFrame([{
    'SSI': ssi,
    'age': age,
    'BMI': bmi,
    'waist_circumference': waist,
    'race': race,
    'hyperlipidemia': hyperlipidemia,
    'diabetes': diabetes
}])

# Make prediction
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]

# Display result
st.subheader("🔍 Infertility Prediction")
if prediction == 1:
    st.error(f"⚠️ Predicted: *Infertile* with probability {probability:.2%}")
else:
    st.success(f"✅ Predicted: *Not Infertile* with probability {1 - probability:.2%}")
