# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
#from flask import Flask
from flask import Flask, request, jsonify
import pickle
import logging
import numpy as np

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/predict',methods=['POST'])
# ‘/’ URL is bound with hello_world() function.
def serve():
    data = request.get_json(force=True)
    prediction = model.predict([[data['age'],
                                          data['hypertension'],
                                          data['heart_disease'],
                                          data['avg_glucose_level'],
                                          data['bmi'],
                                          data['Residence_type_Rural'],
                                          data['Residence_type_Urban'],
                                          data['work_type_Govt_job'],
                                          data['work_type_Never_worked'],
                                          data['work_type_Private'],
                                          data['work_type_Self-employed'],
                                          data['work_type_children'],
                                          data['smoking_status_Unknown'],
                                          data['smoking_status_formerly smoked'],
                                          data['smoking_status_never smoked'],
                                          data['smoking_status_smokes'],
                                          data['ever_married_No'],
                                          data['ever_married_Yes'],
                                          data['gender_Female'],
                                          data['gender_Male'],
                                          data['gender_Other']]])
    print(prediction)
    #return data

    # Make prediction using model loaded from disk as per the data.
    #prediction = model.predict([45,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0])
    #output = prediction[0]
    #return jsonify(output)
    print(type(prediction[0]))
    return '123'

'''
@app.route('/predict',methods=['POST'])
def predict():
    logging.info('123')
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([45,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0])
    output = prediction[0]
    return jsonify(output)
'''
# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()