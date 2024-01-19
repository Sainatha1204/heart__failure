from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load the random forest model
model = joblib.load("random_forest_model.joblib")

# Load the scaler
scaler = joblib.load("standard_scaler.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from the form
            name = request.form['name']
            age = float(request.form['age'])
            creatinine_phosphokinase = float(request.form['creatinine_phosphokinase'])
            ejection_fraction = float(request.form['ejection_fraction'])
            serum_creatinine = float(request.form['serum_creatinine'])
            serum_sodium = float(request.form['serum_sodium'])
            time = float(request.form['time'])
            anaemia = int(request.form['anaemia'])
            diabetes = int(request.form['diabetes'])
            high_blood_pressure = int(request.form['high_blood_pressure'])
            sex = int(request.form['sex'])
            smoking = int(request.form['smoking'])

            # Create a DataFrame with the input values
            input_data = pd.DataFrame({
                'age': [age],
                'creatinine_phosphokinase': [creatinine_phosphokinase],
                'ejection_fraction': [ejection_fraction],
                'serum_creatinine': [serum_creatinine],
                'serum_sodium': [serum_sodium],
                'time': [time],
                'anaemia_0': [1 - anaemia],
                'anaemia_1': [anaemia],
                'diabetes_0': [1 - diabetes],
                'diabetes_1': [diabetes],
                'high_blood_pressure_0': [1 - high_blood_pressure],
                'high_blood_pressure_1': [high_blood_pressure],
                'sex_0': [1 - sex],
                'sex_1': [sex],
                'smoking_0': [1 - smoking],
                'smoking_1': [smoking]
            })

            # Scale the input data using the pre-fitted scaler
            input_data_scaled = scaler.transform(input_data)

            # Make prediction using the random forest model
            prediction = model.predict(input_data_scaled)

            # Create a report based on the prediction result
            report = {
                'name': name,
                'age': age,
                'sex': 'Male' if sex == 1 else 'Female',
                'creatinine_phosphokinase': creatinine_phosphokinase,
                'ejection_fraction': ejection_fraction,
                'serum_creatinine': serum_creatinine,
                'serum_sodium': serum_sodium,
                'time': time,
                'anaemia': 'Yes' if anaemia == 1 else 'No',
                'diabetes': 'Yes' if diabetes == 1 else 'No',
                'high_blood_pressure': 'Yes' if high_blood_pressure == 1 else 'No',
                'smoking': 'Yes' if smoking == 1 else 'No',
                'prediction': 'Has a chance of heart failure' if prediction == 1 else 'No chance of heart failure'
            }

            # Return the report and prediction as JSON
            return render_template('report.html', report=report)

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
