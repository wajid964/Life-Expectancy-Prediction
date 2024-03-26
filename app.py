import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Import the model and preprocessed data
try:
    stacking_pipe = pickle.load(open('stacking_reg_model.pkl', 'rb'))
    df = pickle.load(open('preprocessed_data.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading the model or data: {e}")
    st.stop()

st.title("Life Expectancy Predictor")

# Sample values for population, GDP, and infant deaths
sample_population = 12753375.12
sample_gdp = 7483.158
sample_infant_deaths = 30

# Input elements for user input
country = st.selectbox('Country', df['Country'].unique())
year = st.slider('Year', 2000, 2024, 2019)
status = st.selectbox('Status', df['Status'].unique())
adult_mortality = st.slider('Adult Mortality (per 1000)', min_value=0, max_value=1000, step=1)
infant_deaths = st.slider('Infant Deaths', min_value=0, max_value=1800, step=1, value=sample_infant_deaths)
alcohol = st.slider('Alcohol Consumption (liters)', min_value=0.0, max_value=100.0, step=0.1)
percentage_expenditure = st.slider('Percentage Expenditure (%)', min_value=0.0, max_value=100.0, step=0.1)
hepatitis_b = st.slider('Hepatitis B Coverage (%)', min_value=0.0, max_value=100.0, step=0.1)
measles = st.slider('Measles Cases', min_value=0, max_value=10000, step=1)
bmi = st.slider('BMI', min_value=0.0, max_value=100.0, step=0.1)
under_five_deaths = st.slider('Under-Five Deaths', min_value=0, max_value=500, step=1)
polio = st.slider('Polio Coverage (%)', min_value=0.0, max_value=100.0, step=0.1)
total_expenditure = st.slider('Total Expenditure (%)', min_value=0.0, max_value=100.0, step=0.1)
diphtheria = st.slider('Diphtheria Coverage (%)', min_value=0.0, max_value=100.0, step=0.1)
hiv_aids = st.slider('HIV/AIDS Cases', min_value=0.0, max_value=100.0, step=0.1)
gdp = st.slider('GDP', min_value=0, max_value=120000, step=100, value=int(sample_gdp))
population = st.slider('Population', min_value=0, max_value=1300000000, step=10000, value=int(sample_population))
thinness_1_19_years = st.slider('Thinness 1-19 Years (%)', min_value=0.0, max_value=100.0, step=0.1)
thinness_5_9_years = st.slider('Thinness 5-9 Years (%)', min_value=0.0, max_value=100.0, step=0.1)
income_composition = st.slider('Income Composition of Resources', min_value=0.0, max_value=1.0, step=0.01)
schooling = st.slider('Schooling', min_value=0.0, max_value=100.0, step=0.1)

# Add some empty space above and below the button to center it
st.write("  ")
st.write("  ")
if st.button('Predict Life Expectancy', key='predict_button'):
    try:
        # Prepare input for prediction
        input_data = pd.DataFrame({
            'Country': [country],
            'Year': [year],
            'Status': [status],
            'Life expectancy ': [0],  # Dummy value for the target variable
            'Adult Mortality': [adult_mortality],
            'infant deaths': [infant_deaths],
            'Alcohol': [alcohol],
            'percentage expenditure': [percentage_expenditure],
            'Hepatitis B': [hepatitis_b],
            'Measles ': [measles],
            ' BMI ': [bmi],
            'under-five deaths ': [under_five_deaths],
            'Polio': [polio],
            'Total expenditure': [total_expenditure],
            'Diphtheria ': [diphtheria],
            ' HIV/AIDS': [hiv_aids],
            'GDP': [gdp],
            'Population': [population],
            ' thinness  1-19 years': [thinness_1_19_years],
            ' thinness 5-9 years': [thinness_5_9_years],
            'Income composition of resources': [income_composition],
            'Schooling': [schooling]
        })

        # Drop the dummy target variable
        input_data = input_data.drop('Life expectancy ', axis=1)

        # Make prediction
        predicted_life_expectancy = stacking_pipe.predict(input_data)[0]
        st.title(f"The predicted life expectancy is {predicted_life_expectancy:.2f} years")
    except Exception as e:
        st.error(f"Error predicting life expectancy: {e}")
