from flask import Flask, jsonify, request
import pandas as pd
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model_names = [
    'all',
    'fever',
    'general',
    'hydration',
    'medication',
    'pulse',
    'respiration',
    'skin',
    'caregiver',
    'nnmodel',
]

models = {}

try:
    for name in model_names:
        print(f'Loading module "{name.upper()}": ', end='')
        models[name] = tf.keras.models.load_model('./models/' + name)
        print('OK')
except Exception as e:
    print('FAILED')
    print(e)
    exit()

section_features = {
    'fever': ['ageInMonths', 'temperature', 'feverDuration'],
    'medication': [
        'antibiotics',
        'antibioticsHowMany',
        'antibioticsHowMuch',
        'antipyretic',
        'antipyreticHowMany',
        'antipyreticHowMuch'],
    'hydration': [
        'crying',
        'diarrhea',
        'drinking',
        'lastUrination',
        'skinTurgor',
        'tearsWhenCrying',
        'tongue',
        'vomit-01-No', 'vomit-02-Slight', 'vomit-03-Frequent', 'vomit-04-Yellow', 'vomit-05-5<hours'],
    'respiration': [
        'dyspnea',
        'respiratoryRate',
        'wheezing',
        'ageInMonths'],
    'skin': ['glassTest', 'rash', 'skinColor'],
    'pulse': ['pulse', 'ageInMonths'],
    'general': [
        'awareness-01-Normal', 'awareness-02-SleepyOddOrFeverishNightmares', 'awareness-03-NoReactionsNoAwareness',
        'bulgingFontanelleMax18MOld',
        'exoticTrip',
        'lastTimeEating',
        'pain-01-No', 'pain-02-FeelingBad', 'pain-03-Headache', 'pain-04-SwollenPainful', 'pain-05-StrongBellyacheAche',
        'painfulUrination',
        'seizure',
        'smellyUrine',
        'vaccinationIn14days',
        'vaccinationHowManyHoursAgo',
        'wryNeck'],
    'caregiver': [
        'caregiverConfident',
        'caregiverFeel',
        'caregiverThink',
    ],
}


PATIENT_STATE = ['good', 'caution', 'danger']


@app.route("/", methods=['POST'])
def root():
    data = request.get_json()

    print('DATA RECEIVED:', data.get('row'))
    df = pd.DataFrame.from_dict([data.get('row')])

    res = {}
    # Section predictions
    for key, features in section_features.items():
        data = df[features]
        row = np.array(data.to_numpy(), dtype=np.float32)
        state = np.argmax(models[key].predict(row, verbose=0)[0])

        res[key] = PATIENT_STATE[state]
    # PatientState prediction
    row = np.array(df.drop(columns=section_features['caregiver']).to_numpy(), dtype=np.float32)
    state = np.argmax(models['all'].predict(row, verbose=0)[0])
    res['patientState'] = PATIENT_STATE[state]

    print('PREDICTIONS:', res)
    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True)
