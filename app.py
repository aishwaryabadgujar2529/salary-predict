import streamlit as st
import pandas as pd
import pickle

# Load the trained model and imputer
try:
    with open('random_forest_regressor_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('simple_imputer.joblib', 'rb') as imputer_file:
        imputer = pickle.load(imputer_file)
except FileNotFoundError:
    st.error("Error: Model or imputer file not found. Please ensure 'random_forest_regressor_model.pkl' and 'simple_imputer.joblib' are in the same directory.")
    st.stop()

st.title('Salary Prediction App')
st.write('Predict an employee\'s salary based on their characteristics.')

# Input fields for user
age = st.slider('Age', 20, 65, 30)
years_of_experience = st.slider('Years of Experience', 0, 40, 5)

gender_options = {'Male': 1, 'Female': 0}
gender_selection = st.radio('Gender', list(gender_options.keys()))
gender = gender_options[gender_selection]

education_options = {
    'High School': 0,
    'Associate Degree': 1,
    'Bachelor\'s Degree': 2,
    'Master\'s Degree': 3,
    'PhD': 4,
    'Doctorate': 5
}
education_level_selection = st.selectbox('Education Level', list(education_options.keys()))
education_level = education_options[education_level_selection]

# Note: Job Title is categorical and was label encoded. For a real app,
# you\'d need a mapping from actual job titles to their encoded integers.
# For simplicity, we\'ll use a numerical input here.
# A more robust solution would involve providing a selectbox with actual job titles
# and then mapping them to their encoded integers.
job_title = st.number_input('Job Title (Encoded, e.g., 18 for Software Engineer)', min_value=0, max_value=200, value=18)

if st.button('Predict Salary'):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Age': [float(age)],
        'Years of Experience': [float(years_of_experience)],
        'Gender': [float(gender)],
        'Education Level': [float(education_level)],
        'Job Title': [float(job_title)]
    })

    # Impute missing values (even if none, the imputer expects the same transform)
    input_data_imputed = imputer.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_imputed)

    st.success(f'Predicted Salary: ${prediction[0]:,.2f}')
