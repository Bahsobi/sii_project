pip install matplotlib streamlit numpy pandas scikit-learn

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# App title
st.title('🐧 Penguin Species Prediction App')
st.markdown("""
This app predicts penguin species using Random Forest classification!
""")

# Load data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Bahsobi/sii_project/refs/heads/main/50_Startups%20(1).csv'
    return pd.read_csv(url)

df = load_data()

# Display raw data
with st.expander('View Raw Data'):
    st.dataframe(df)

# Data visualization
st.subheader('Data Visualization')
fig, ax = plt.subplots()
ax.scatter(df['bill_length_mm'], df['body_mass_g'], c=LabelEncoder().fit_transform(df['species']))
plt.xlabel('Bill Length (mm)')
plt.ylabel('Body Mass (g)')
st.pyplot(fig)

# Sidebar for user input
with st.sidebar:
    st.header('Input Features')
    
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.selectbox('Sex', ('male', 'female'))
    bill_length = st.slider('Bill length (mm)', 
                           float(df['bill_length_mm'].min()), 
                           float(df['bill_length_mm'].max()), 
                           float(df['bill_length_mm'].mean()))
    bill_depth = st.slider('Bill depth (mm)', 
                          float(df['bill_depth_mm'].min()), 
                          float(df['bill_depth_mm'].max()), 
                          float(df['bill_depth_mm'].mean()))
    flipper_length = st.slider('Flipper length (mm)', 
                              float(df['flipper_length_mm'].min()), 
                              float(df['flipper_length_mm'].max()), 
                              float(df['flipper_length_mm'].mean()))
    body_mass = st.slider('Body mass (g)', 
                         float(df['body_mass_g'].min()), 
                         float(df['body_mass_g'].max()), 
                         float(df['body_mass_g'].mean()))

# Prepare data for modeling
def preprocess_data(df):
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['island', 'sex'])
    
    # Encode target variable
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    
    return df, le

# Model training and prediction
def train_and_predict():
    # Prepare data
    processed_df, label_encoder = preprocess_data(df.copy())
    
    X = processed_df.drop('species', axis=1)
    y = processed_df['species']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.success(f'Model Accuracy: {accuracy:.2%}')
    
    return model, label_encoder

# Create input dataframe
input_data = pd.DataFrame({
    'island': [island],
    'bill_length_mm': [bill_length],
    'bill_depth_mm': [bill_depth],
    'flipper_length_mm': [flipper_length],
    'body_mass_g': [body_mass],
    'sex': [sex]
})

# Combine with original data for consistent preprocessing
full_data = pd.concat([input_data, df.drop('species', axis=1)], axis=0)

# Preprocess the combined data
processed_data, _ = preprocess_data(full_data)
input_processed = processed_data[:1]  # Get just the user input row

# Train model and make prediction
if st.button('Predict Species'):
    model, label_encoder = train_and_predict()
    
    # Make prediction
    prediction = model.predict(input_processed)
    prediction_proba = model.predict_proba(input_processed)
    
    # Display results
    species = label_encoder.inverse_transform(prediction)[0]
    st.success(f'Predicted Species: **{species}**')
    
    # Show prediction probabilities
    prob_df = pd.DataFrame({
        'Species': label_encoder.classes_,
        'Probability': prediction_proba[0]
    })
    
    st.subheader('Prediction Probabilities')
    st.dataframe(prob_df.style.highlight_max(axis=0))
