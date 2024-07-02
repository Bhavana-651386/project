# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['age'],
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
                                          data['gender_Other'])]])
    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)