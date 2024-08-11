import streamlit as st
from utils import PrepProcesor, columns 

import numpy as np
import pandas as pd
import joblib

model = joblib.load('xgbpipe.joblib')
st.title('Will you survive if you were among Titanic passengers or not :ship:')
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
pasengerid = st.text_input('Give Us Your PasangerId Please' , '1414')
p_class = st.selectbox('Give Us Your Class Please' , [1,2,3])
name = st.text_input('Give Us Your Name Please' , 'Golnoush ahmadi')
sex= st.select_slider('Choose Your Gender Please' , ['Male' , 'Female'])
age = st.slider('Give Us Your Age Please' , 0,100)
sibsp = st.slider('Choose Your The Number Of Your Siblings Or Spouse Please' , 0,10)
parch = st.slider('Choose Your Parch Please' , 0,10)
ticket = st.text_input('Give Us Your Ticket Number Please' , '1414')
fare = st.number_input('Give Us Your Fare Price Please' , 0,1000)
cabin= st.text_input('Choose Your cabin Please' , 'C52')
embarked = st.select_slider('Did They Embark?' , ['S' , 'C' , 'Q'])
def predict(): 
    row = np.array([pasengerid ,p_class,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('Passenger Survived :thumbsup:')
    else: 
        st.error('Passenger did not Survive :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)

