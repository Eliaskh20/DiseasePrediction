from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)

#Elias Alkharma
svm_model = joblib.load('svmBloodGlucose_model.pkl')
blood_glucose_model = joblib.load('BloodGlucose_model.pkl')
scaler = joblib.load('scalerBloodGlucose.pkl')
selector = joblib.load('selectorBloodGlucose.pkl')

#dema alasaad
knn_model = joblib.load('HRknn_model.pkl')
HRScaler = joblib.load('HRScaler.pkl')

#gharam alabdalah

# Load the model
with open('GHBloodGlucose.sav', 'rb') as file:
    Gh_blood_glucose_model = pickle.load(file)


#Nayaz
 # Load the saved model
    loaded_model = pickle.load(open('heart_disease_pred.sav', 'rb'))



#Amaar
# Load the models and feature names
with open('Parkinsson_model.pkl', 'rb') as file:
    parkinsson_model = pickle.load(file)


#Ayman
# Load the thyroid disease detection model
with open('thyroid_disease_detection.sav', 'rb') as file:
    thyroid_model = joblib.load(file)

@app.route('/')
def index():
    return render_template('index.html')



def validate_input(data):
    required_fields = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    for field in required_fields:
        if field not in data:
            return False, f'Missing field: {field}'
        if not isinstance(data[field], (int, float)):
            return False, f'Invalid type for field: {field}'
    return True, ''

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        # Your existing logic here
        data = request.json
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        try:
            input_features = np.array([[
                data['Pregnancies'],
                data['Glucose'],
                data['BloodPressure'],
                data['SkinThickness'],
                data['Insulin'],
                data['BMI'],
                data['DiabetesPedigreeFunction'],
                data['Age']
            ]])
            # SVM prediction
            svm_prediction = svm_model.predict(input_features)[0]

            # Preprocessing and BloodGlucose model prediction
            scaled_features = scaler.transform(input_features)
            selected_features = selector.transform(scaled_features)
            blood_glucose_prediction = blood_glucose_model.predict(selected_features)[0]

            return jsonify({
                'svm_prediction': int(svm_prediction),
                'blood_glucose_prediction': int(blood_glucose_prediction)
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Render the HTML form on GET requests
    return render_template('page1.html')


@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if request.method == 'POST':
        data = request.json
        input_data = np.array([[
            int(data.get('age', 0)),
            int(data.get('sex', 0)),
            int(data.get('chest_pain_type', 0)),
            int(data.get('bp', 0)),
            int(data.get('cholesterol', 0)),
            int(data.get('fbs_over_120', 0)),
            int(data.get('ekg_results', 0)),
            int(data.get('max_hr', 0)),
            int(data.get('exercise_angina', 0)),
            float(data.get('st_depression', 0.0)),
            int(data.get('slope_of_st', 0)),
            int(data.get('number_of_vessels_fluro', 0)),
            int(data.get('thallium', 0))
        ]])

        # Apply the HRScaler to the input data
        scaled_data = HRScaler.transform(input_data)

        # Make predictions using the KNN model
        prediction = knn_model.predict(scaled_data)[0]
        return jsonify({'knn_prediction': int(prediction)})

    return render_template('page2.html')


@app.route('/page3', methods=['GET', 'POST'])
def page3():
    if request.method == 'POST':
        data = request.json
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        try:
            input_features = np.array([[
                data['Pregnancies'],
                data['Glucose'],
                data['BloodPressure'],
                data['SkinThickness'],
                data['Insulin'],
                data['BMI'],
                data['DiabetesPedigreeFunction'],
                data['Age']
            ]])
            # GH Blood Glucose Model prediction
            blood_glucose_prediction = Gh_blood_glucose_model.predict(input_features)[0]

            return jsonify({
                'GH_blood_prediction': int(blood_glucose_prediction),
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Render the HTML form on GET requests
    return render_template('page3.html')



@app.route('/page4', methods=['GET', 'POST'])
def page4():
    if request.method == 'POST':
        data = request.json
        try:
            input_data = [
                int(data.get('Age', 0)),
                int(data.get('Sex', 0)),
                int(data.get('ChestPainType', 0)),
                int(data.get('RestingBP', 0)),
                int(data.get('Cholesterol', 0)),
                int(data.get('FastingBS', 0)),
                int(data.get('RestingECG', 0)),
                int(data.get('MaxHR', 0)),
                int(data.get('ExerciseAngina', 0)),
                float(data.get('Oldpeak', 0.0)),
                int(data.get('ST_Slope', 0))
            ]

            # Make prediction
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            prediction = loaded_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                diagnosis = 'The person does not have heart disease.'
            else:
                diagnosis = 'The person has heart disease.'

            return jsonify({'diagnosis': diagnosis})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('page4.html')




@app.route('/page5', methods=['GET', 'POST'])
def page5():
    if request.method == 'POST':
        data = request.json
        input_data = [
            int(data.get('MDVP:Fo(Hz)', 0)),
            int(data.get('MDVP:Fhi(Hz)', 0)),
            int(data.get('MDVP:Flo(Hz)', 0)),
            float(data.get('MDVP:Jitter(%)', 0.00)),
            float(data.get('Jitter:DDP', 0.00)),
            float(data.get('Shimmer:APQ5', 0.00)),
            int(data.get('HNR', 0)),
            float(data.get('spread1', 0.00)),
            float(data.get('spread2', 0.00)),
            float(data.get('PPE', 0.00)),

        ]
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = parkinsson_model.predict(input_data_reshaped)
        if prediction == 1:
            result = 'Have Parkinson'
        elif prediction == 0:
            result = 'Do not have Parkinson'
        else:
            result = 'An error occurred: ' + prediction

        return jsonify({'Parkinson': result})

    return render_template('page5.html')



@app.route('/page6', methods=['GET', 'POST'])
def page6():
    if request.method == 'POST':
        data = request.json
        try:
            input_features = pd.DataFrame({
                'age': [int(data.get('age', 0))],
                'sex': [int(data.get('sex', 0))],
                'TT4': [float(data.get('TT4', 0.0))],
                'T3': [float(data.get('T3', 0.0))],
                'T4U': [float(data.get('T4U', 0.0))],
                'FTI': [float(data.get('FTI', 0.0))],
                'TSH': [float(data.get('TSH', 0.0))],
                'pregnant': [int(data.get('pregnant', 0))]
            })

            # Make predictions using the thyroid disease detection model
            input_data_as_numpy_array2 = np.asarray(input_features)
            input_data_reshaped2 = input_data_as_numpy_array2.reshape(1, -1)
            prediction = thyroid_model.predict(input_features)[0]

            if prediction == 1:
                resul = 'Unfortunately, the patient suffers from hypothyroidism'
            elif prediction == 2:
                resul = 'The patient is in normal condition and does not suffer from any thyroid problems.'
            else:
                resul = 'Unfortunately, the patient suffers from hyperthyroidism.'

            return jsonify({'diagnosis': resul})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('page6.html')


if __name__ == '__main__':
    app.run(debug=True)
