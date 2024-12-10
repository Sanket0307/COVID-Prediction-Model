from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('covid_symptom_prediction_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    fever = int(request.form.get('fever', 0))
    tiredness = int(request.form.get('tiredness', 0))
    dry_cough = int(request.form.get('dry_cough', 0))
    difficulty_breathing = int(request.form.get('difficulty_breathing', 0))
    sore_throat = int(request.form.get('sore_throat', 0))
    none_symptom = int(request.form.get('none_symptom', 0))
    pains = int(request.form.get('pains', 0))
    nasal_congestion = int(request.form.get('nasal_congestion', 0))
    runny_nose = int(request.form.get('runny_nose', 0))
    diarrhea = int(request.form.get('diarrhea', 0))
    none_experiencing = int(request.form.get('none_experiencing', 0))
    severity_mild = int(request.form.get('severity_mild', 0))
    severity_moderate = int(request.form.get('severity_moderate', 0))
    severity_none = int(request.form.get('severity_none', 0))
    severity_severe = int(request.form.get('severity_severe', 0))
    contact_dont_know = int(request.form.get('contact_dont_know', 0))
    contact_no = int(request.form.get('contact_no', 0))
    contact_yes = int(request.form.get('contact_yes', 0))

    # Age and Gender selections
    age_0_9 = int(request.form.get('age_0_9', 0))
    age_10_19 = int(request.form.get('age_10_19', 0))
    age_20_24 = int(request.form.get('age_20_24', 0))
    age_25_59 = int(request.form.get('age_25_59', 0))
    age_60_plus = int(request.form.get('age_60_plus', 0))
    gender_female = int(request.form.get('gender_female', 0))
    gender_male = int(request.form.get('gender_male', 0))
    gender_transgender = int(request.form.get('gender_transgender', 0))

    # Prepare input for prediction
    input_data = [
        fever, tiredness, dry_cough, difficulty_breathing, sore_throat,
        none_symptom, pains, nasal_congestion, runny_nose, diarrhea,
        none_experiencing, severity_mild, severity_moderate, severity_none,
        severity_severe, contact_dont_know, contact_no, contact_yes,
        age_0_9, age_10_19, age_20_24, age_25_59, age_60_plus,
        gender_female, gender_male, gender_transgender
    ]

    # Predict
    prediction = model.predict([input_data])
    result = "Positive for COVID-19" if prediction[0] == 1 else "Negative for COVID-19"

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
