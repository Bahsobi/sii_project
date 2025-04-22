import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

st.title('💼 Startup Profit Predictor')

st.info('Predict the **Profit** based on startup data using Multiple Linear Regression.')

# Load data
df = pd.read_csv('/mnt/data/50_Startups (1).csv')

with st.expander('Data Overview'):
    st.write(df)

# Preprocess X and y
X_raw = df.drop('Profit', axis=1)
y = df['Profit']

# Encode the 'State' column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['State'])], remainder='passthrough')
X_encoded = ct.fit_transform(X_raw)

# Train the model
regressor = LinearRegression()
regressor.fit(X_encoded, y)

# Input section
with st.sidebar:
    st.header('Enter Startup Details')

    state = st.selectbox('State', df['State'].unique())
    rnd_spend = st.number_input('R&D Spend', value=0.0)
    admin = st.number_input('Administration', value=0.0)
    marketing = st.number_input('Marketing Spend', value=0.0)

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[state, rnd_spend, admin, marketing]],
                              columns=['State', 'R&D Spend', 'Administration', 'Marketing Spend'])

    # Encode input
    input_encoded = ct.transform(input_data)

# Make prediction
prediction = regressor.predict(input_encoded)

# Display result
st.subheader('📈 Predicted Profit')
st.success(f"${prediction[0]:,.2f}")
