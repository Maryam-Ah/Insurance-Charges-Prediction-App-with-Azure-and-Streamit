import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import joblib 
import pickle
import math
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
import json
import logging
import os






from PIL import Image
image = Image.open('image.jpg')
st.sidebar.image(image,use_column_width=True)



@st.cache
def predict(model, input_df):
    predictions= model.predict(input_df)
    return predictions

categorical_features = ['sex', 'smoker', 'region']
numerical_features= ['age', 'bmi', 'children']

st.title("Insurance Charges Prediction App")
age = st.number_input('Age', min_value=1, max_value=100, value=25)
sex = st.radio("Sex",("male","female"),key='sex')
# sex = st.selectbox('Sex', ['male', 'female'])
# bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
bmi = st.slider("Bmi", 10,50,key='bmi')
children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
if st.checkbox('Smoker'):
    smoker = 'yes'
else:
    smoker = 'no'
    
region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])
input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
input_df = pd.DataFrame([input_dict],index=np.arange(len(input_dict)),columns = categorical_features + numerical_features)







add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Random Forest Regressior", "Support Vector Regression","GradientBoosting"))








# output=[]
output=""

    
    
# Load the data prepration pipline back from file
Pkl_Filename = "data_prepration.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    Pickled_data_prepration = pickle.load(file)

# ...................................................
if add_selectbox == 'Random Forest Regressior':

    
     # Load the Model back from file
    filename = 'rf.pkl'
    with open(filename, 'rb') as file:  
        model = pickle.load(file)

                       
                        
    if st.button("Predict"):
            data_prep = Pickled_data_prepration.transform(input_df)
            output = predict(model=model, input_df=data_prep)
            output = '$' + str(math.trunc(output[0]))


    st.success('The output is : {}'.format(output))
    
# ..........................................
if add_selectbox == 'Support Vector Regression':
    
    filename = 'svr.pkl'
    with open(filename, 'rb') as file:  
        model = pickle.load(file)

                      
                        
    if st.button("Predict"):
            data_prep = Pickled_data_prepration.transform(input_df)
            output = predict(model=model, input_df=data_prep)
            output = '$' + str(math.trunc(output[0]))


    st.success('The output is : {}'.format(output))


    
# ..........................................
if add_selectbox == 'GradientBoosting':
    
    filename = 'gb.pkl'
    with open(filename, 'rb') as file:  
        model = pickle.load(file)

                      
                        
    if st.button("Predict"):
            data_prep = Pickled_data_prepration.transform(input_df)
            output = predict(model=model, input_df=data_prep)
            output = '$' + str(math.trunc(output[0]))


    st.success('The output is : {}'.format(output))


    
    
    
    
    
    
# if __name__ == '__main__':
#     run()
