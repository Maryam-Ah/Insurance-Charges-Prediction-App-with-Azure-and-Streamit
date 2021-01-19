import os
import math
import json
import logging
import pickle
import joblib 
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import seaborn as sns
from sklearn.svm import SVR
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error ,mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer


# load the image
image = Image.open('image-smile.png').convert('RGB')
# st.sidebar.image(image,use_column_width=True)
st.title("Insurance Charges Prediction App")


Prediction = st.radio("Choose the action: ",("Predict on the training data","Predict on a new data"))


if Prediction == 'Predict on a new data':

    # prediction function
    @st.cache
    def predict(model, input_df):
        predictions= model.predict(input_df)
        return predictions

    categorical_features = ['sex', 'smoker', 'region']
    numerical_features= ['age', 'bmi', 'children']


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


        
        
        
        
        
 # ......................................................................................................... 
 # .........................................................................................................  
 # .........................................................................................................  
 # .........................................................................................................  
        
if Prediction == 'Predict on the training data':

    
    @st.cache(persist=True)
    def load_data():
        insurance_df=pd.read_csv('nsurance.csv')
        return insurance_df
    
    
    
    @st.cache(persist=True)
    def split(insurance_df):
        insurance = insurance_df.drop(['charges'], axis=1) # drop labels for training set
        insurance_labels = insurance_df['charges'].copy()
        
        # splitting train and test data
        X_train, X_test, y_train, y_test = train_test_split(insurance, insurance_labels, test_size=0.4)
        X_train_num= X_train.drop([ 'sex','smoker','region'],axis=1)
        X_train_cat = X_train[['sex', 'smoker','region']]
        num_attribs = list(X_train_num)
        cat_attribs = ['sex', 'region','smoker']
        return X_train,X_test,y_train,y_test,X_train_num,X_train_cat,num_attribs,cat_attribs
        
    
    
    class DataFrameSelector( BaseEstimator, TransformerMixin ):
        #Class Constructor 
        def __init__( self, feature_names ):
            self.feature_names = feature_names 

        #Return self nothing else to do here    
        def fit( self, X, y = None ):
            return self 

        #Method that describes what we need this transformer to do
        def transform( self, X, y = None ):
    #         return X[ self.feature_names ].values
              df = X.copy()
            # convert columns to categorical
              for name in df.columns.to_list():
                    col = pd.Categorical(df[name])
                    df[name] = col.codes

            #returns numpy array
              return df
    
        
        df = load_data()
        X_train,X_test,y_train,y_test,X_train_num,X_train_cat,num_attribs,cat_attribs = split(df)
        
        
        
        num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

        cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        full_pipeline = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_attribs),
            ('cat', cat_pipeline, cat_attribs)])

      
    
        st.sidebar.subheader("How would you like to predict?")
        Regressior = st.sidebar.selectbox("Regressior alforithm:",("Random Forest Regressior","Support Vector Regression","Gradient Boosting"))
        
        
        
        if st.sidebar.checkbox("Show the size of Data set ,Training set and test set ",False):
            st.write('Data set shape',df.shape)
            st.write('X_train shape',X_train.shape)
            st.write('X_test shape',X_test.shape)
            st.write('y_train shape',y_train.shape)
            st.write('y_test shape',y_test.shape)
            
            
            
        if st.sidebar.checkbox("Show training/raw data ",False):
            st.subheader("Insurance Data Set (Regression)")
            st.write(df)
            
            
            
        #to see the relationship between the training data values
        if st.sidebar.checkbox("Plot the relationship between features ",False):
            sns.pairplot(df)
            st.pyplot()

         # ...................................................
        if Regressior == 'Random Forest Regressior':
            st.subheader("Model Hyperparameters (Regularization parameter)")
            # n_lowest_estimators = st.number_input("Lowest number of estimators (between 2 -30)", 2,30,step=1,key='n_lowest_estimators')
            n_highest_estimators = st.number_input("Number of estimators (between 2 -31) ", 3,31,step=1,key='n_highest_estimators')
            
            # n_lowest_features = st.number_input("Lowest number of features (between 1 -3)", 1,3,step=1,key='n_lowest_features')
            n_highest_features = st.number_input("Number of features (between 2 -4)", 2,4,step=1,key='n_highest_features')
            
            param_distribs = {
                    'n_estimators': randint(low=1, high=n_highest_estimators),
                    'max_features': randint(low=1, high=n_highest_features),
                }

            forest_reg = RandomForestRegressor(random_state=42)
            rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                            n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42)
            
            
            X_train_prepared = full_pipeline.fit_transform(X_train)
            rnd_search.fit(X_train_prepared, y_train)
            
            if st.button("Predict"):
                y_train_pred= rnd_search.predict(X_train_prepared)
                rf_mse = mean_squared_error(y_train, y_train_pred)
                rf_mae = mean_absolute_error(y_train, y_train_pred)
                rf_rmse = np.sqrt(rf_mse)
                st.success('Root Mean Squared Error : {}'.format(math.trunc(rf_rmse)))
                st.success('Mean Absolute Error / L1 Loss : {}'.format(math.trunc(rf_mae)))
            
            
            
         # ...................................................
        if Regressior == 'Support Vector Regression':
            st.subheader("Model Hyperparameters")
            C = st.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C')
            epsilon = st.slider("Epsilon (Regularization parameter)", 0.01,1.0,key='epsilon')
            kernel = st.radio("Kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"),key = 'kernel')
            gamma = st.radio("Gamma (Kernel Cofficient)",("scale","auto"),key='gamma')
            svr = SVR(C=C, epsilon=epsilon, gamma = gamma, kernel=kernel)
            X_train_prepared_svr = full_pipeline.fit_transform(X_train)
            svr.fit(X_train_prepared_svr, y_train)
            if st.button("Predict"):
                y_train_pred_svr= svr.predict(X_train_prepared_svr)
                svr_mse = mean_squared_error(y_train, y_train_pred_svr)
                svr_mae = mean_absolute_error(y_train, y_train_pred_svr)
                svr_rmse = np.sqrt(svr_mse)
                st.success('Root Mean Squared Error  : {}'.format(math.trunc(svr_rmse)))
                st.success('Mean Absolute Error / L1 Loss : {}'.format(math.trunc(svr_mae)))
            
        # ...................................................
        if Regressior == 'Gradient Boosting':
            
            st.subheader("Model Hyperparameters (Regularization parameter)")
            n_lowest_estimators = st.number_input("Lowest number of estimators ", 4,40,step=1,key='n_lowest_estimators')
            n_highest_estimators = st.number_input("Highest number of estimators ", 5,40,step=1,key='n_highest_estimators')
            
            
            
            param_distribs = {
        'n_estimators': randint(low=n_lowest_estimators, high=n_highest_estimators)
        #         'loss':loss(‘ls’, ‘lad’, ‘huber’, ‘quantile’),
        #         'learning_rate': randint(low=0.001, high=0.1),
            }

            
            gb = GradientBoostingRegressor()
            gb_search = RandomizedSearchCV(gb, param_distributions=param_distribs,
                                            n_iter=5, cv=10, scoring='neg_mean_squared_error', random_state=42)



            X_train_prepared_gb = full_pipeline.fit_transform(X_train)
            gb_search.fit(X_train_prepared_gb, y_train)
            if st.button("Predict"):
                y_train_pred_gb= gb_search.predict(X_train_prepared_gb)
                gb_mse = mean_squared_error(y_train, y_train_pred_gb)
                gb_mae = mean_absolute_error(y_train, y_train_pred_gb)
                gb_rmse = np.sqrt(gb_mse)
                st.success('Root Mean Squared Error : {}'.format(math.trunc(gb_rmse)))
                st.success('Mean Absolute Error / L1 Loss : {}'.format(math.trunc(gb_mae)))
         

