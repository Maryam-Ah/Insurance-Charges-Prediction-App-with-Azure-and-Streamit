import pandas as pd
import numpy as np
import streamlit as st
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import joblib 
import pickle


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV





def predict(model, input_df):
    predictions= model.predict(input_df)
    return predictions


st.title("Insurance Charges Prediction App")
age = st.number_input('Age', min_value=1, max_value=100, value=25)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
if st.checkbox('Smoker'):
    smoker = 'yes'
else:
    smoker = 'no'
region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])    
input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
input_df = pd.DataFrame([input_dict])


output=[]
# output=""
# Load the Model back from file
filename = 'rfr.pkl'
with open(filename, 'rb') as file:  
      model = pickle.load(file)

        
        
st.subheader('Class Labels and their corresponding index number')
label_name = np.array(['Age',
    'Sex','BMI','Children','Smoker'])

st.write(label_name)                        
                        
if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = int(output[0])


st.success('The output is : {}'.format(label_name[output]))
    




# if __name__ == '__main__':
#     run()
