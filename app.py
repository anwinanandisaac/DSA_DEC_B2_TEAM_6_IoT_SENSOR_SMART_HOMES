from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import statistics as stat


app=Flask(__name__)
with open('model_LVL6_UP_2023-2024_dec_B_Team_06.pkl','rb') as model_file:
    model=pickle.load(model_file)
                      
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   icon_stat=request.form['icon_status']
   
   if icon_stat=="clear-day":
     icon_status=0
   elif icon_stat=="clear-night":
     icon_status=1
   elif icon_stat=="cloudy":
     icon_status=2
   elif icon_stat=="fog":
     icon_status=3
   elif icon_stat=="partly-cloudy-day":
     icon_status=4
   elif icon_stat=="partly-cloudy-night":
     icon_status=5
   elif icon_stat=="rain":
     icon_status=6
   elif icon_stat=="snow":
     icon_status=7
   elif icon_stat=="wind":
     icon_status=8
   
   use_KW=float(request.form['use'])
   gen_KW=float(request.form['gen'])
   Dishwasher_kW=float(request.form['Dishwasher'])
   Furnace_1_kW=float(request.form['Furnace 1'])
   Furnace_2_kW=float(request.form['Furnace 2'])
   Home_office_kW=float(request.form['Home office'])
   Fridge_kW=float(request.form['Fridge'])
   Wine_cellar_kW=float(request.form['Wine cellar'])
   Garage_door_kW=float(request.form['Garage door'])
   Kitchen_12_kW=float(request.form['Kitchen 12'])
   Kitchen_14_kW=float(request.form['Kitchen 14'])
   Kitchen_38_kW=float(request.form['Kitchen 38'])
   Barn_kW=float(request.form['Barn'])
   Well_kW=float(request.form['Well'])
   Microwave_kW=float(request.form['Microwave'])
   Living_room_kW=float(request.form['Living room'])
   temperature=float(request.form['temperature'])
   humidity=float(request.form['humidity'])
   visibility=float(request.form['visibility'])
   pressure=float(request.form['pressure'])
   windSpeed=float(request.form['windSpeed'])
   cloudCover=float(request.form['cloudCover'])
   windBearing=float(request.form['windBearing'])
   precipIntensity=float(request.form['precipIntensity'])
   precipProbability=float(request.form['precipProbability'])
  
   
   




   feature=np.array([[use_KW,gen_KW,Dishwasher_kW,Furnace_1_kW,Furnace_2_kW,Home_office_kW,Fridge_kW,Wine_cellar_kW,Garage_door_kW,Kitchen_12_kW,Kitchen_14_kW,Kitchen_38_kW,Barn_kW,Well_kW,Microwave_kW,Living_room_kW,temperature,icon_status,humidity,visibility,pressure,windSpeed,cloudCover,windBearing,precipIntensity,precipProbability]])
  


  
   
   prediction=model.predict(feature)
   return render_template('index.html',pred_res=prediction[0])

   
if __name__=='__main__':
  app.run(debug=True) 
