"""
Created on Tue Jan 30 08:15:53 2024

@author: Sk. Atahar Ali
"""
import streamlit as st
import pickle
import pandas as pd
from src.utils import load_object

# Load ML Model
preprocessed_obj = load_object('artifacts/pre_processed_obj/pre_processed_obj.pkl')
base_model = load_object("model\model.pkl")
#base_model = pickle.load(open('model.pkl','rb'))
#------------------------------------------ Streamlit App ------------------------------------------#
# Page Configuration
st.set_page_config(
     page_title="Student Score Predictor",
     page_icon="🧊",
     layout="wide", #   centered
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://github.com/ataharsk10/ml_regression',
         'Report a bug': "https://github.com/ataharsk10/ml_regression",
         'About': "Student Score Predictor. @author: *Sk. Atahar Ali* "
     }
 )

#----- App Header
st.markdown("""<h1 style="color:#3399ff;font-size:40px;">Student Score Predictor</h1>""", unsafe_allow_html=True)
st.markdown("""<hr style="height:4px;border:none;color:#3399ff;background-color:#3399ff;" /> """, unsafe_allow_html=True)

if 'key' not in st.session_state:
    st.session_state['result'] = 0.0

# Section for taking input values for features from user.
with st.container():
    st.markdown("""<h6 style="color:#4da6ff;font-size:30px;">Enter Inputs</h6>""", unsafe_allow_html=True)
    #st.caption('Enter Inputs')
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col2:
        input_gender = st.selectbox(label = 'Gender', options =('male', 'female'), index = None)
        input_lunch = st.selectbox(label = 'Lunch', options =('free/reduced', 'standard'), index = None)
        input_reading_score = st.number_input(label='Reading Score',min_value=None, max_value=None, value="min")

    with col3:
        input_race_ethnicity = st.selectbox(label = 'Race Ethnicity', options =('group A', 'group B', 'group C', 'group D', 'group E' ), index = None)
        input_test_preparation_course = st.selectbox(label = 'Test Preparation Course', options =('none', 'completed'), index = None)
        input_test_writing_score = st.number_input(label='Test Writing Score',min_value=None, max_value=None, value="min")

    with col4:
        input_parental_level_of_education = st.selectbox(label = 'Parental Level Of Education', options =("associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"), index = None)
        input_test_math_score = st.number_input(label='Test Math Score',min_value=None, max_value=None, value="min")

with st.container():
    col1, col2, col3, col4, col5 = st.columns(5)
    with col3:
        st.markdown("""</br>""",unsafe_allow_html=True) #Spacing
        if st.button('Submit Input & Predict'): #(Section-A-4)
            input_to_predict = {
                'gender': input_gender,
                'race_ethnicity': input_race_ethnicity,
                'parental_level_of_education': input_parental_level_of_education,
                'lunch': input_lunch,
                'test_preparation_course': input_test_preparation_course,
                'math_score': input_test_math_score,
                'reading_score': input_reading_score,
                'writing_score': input_test_writing_score
                }

            # Prediction
            test_input_df = pd.DataFrame(input_to_predict,index=[0])
            transform_data = preprocessed_obj.transform(test_input_df)
            predicted_score = base_model.predict(transform_data) ##
            st.session_state['result'] = predicted_score[0]

        st.text_input(label='Predicted Score ',value = st.session_state.result)
    


