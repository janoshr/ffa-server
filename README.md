# Fever Friend app server

Flask server acting as an API for the machine learning module of the FeverFellow mobile application.

Installing required packages for python. Recommended python version is 3.10 (this version was used in development).

```bash
pip install -r requirements.txt
```

To start the sever execute the following command.

```bash
python3 app.py
```

## Models

The Tensorflow models should in a `models` folder under the project root. The models are called `all`, `caregiver`, `fever`, `general`, `hydration`, `medication`, `nnmodel`, `pulse`, `respiration`, `skin`. Most models are section models except for `all` and `nnmodel` these are models to predict the overall condition of the patient, one with random forest model the other with artificial neural networks, respectively.

The models should be in `SavedModel` format of TensorFlow.

The API has only one route `/` that is a `POST` request. It expects a JSON object of the format:

```json
{
    "row": {
        "ageInMonths": 15.6173706445342,
        "feverDuration": 1.0,
        "temperature": 38.3,
        "antibiotics": 0.0,
        "antibioticsHowMany": 0.0,
        "antibioticsHowMuch": 0.0,
        "antipyretic": 1.0,
        "antipyreticHowMany": 1.0,
        "antipyreticHowMuch": 50.0,
        "crying": 0.0,
        "diarrhea": 0.0,
        "drinking": 0.0,
        "lastUrination": 0.0,
        "skinTurgor": 0.0,
        "tearsWhenCrying": 0.0,
        "tongue": 0.0,
        "dyspnea": 1.0,
        "respiratoryRate": 13.0,
        "wheezing": 0.0,
        "glassTest": 0.0,
        "rash": 0.0,
        "skinColor": 0.0,
        "pulse": 68.0,
        "bulgingFontanelleMax18MOld": 0.0,
        "exoticTrip": 0.0,
        "lastTimeEating": 0.0,
        "painfulUrination": 0.0,
        "seizure": 0.0,
        "smellyUrine": 0.0,
        "vaccinationIn14days": 0.0,
        "vaccinationHowManyHoursAgo": 0.0,
        "wryNeck": 0.0,
        "pain-01-No": 1.0,
        "pain-02-FeelingBad": 0.0,
        "pain-03-Headache": 0.0,
        "pain-04-SwollenPainful": 0.0,
        "pain-05-StrongBellyacheAche": 0.0,
        "awareness-01-Normal": 1.0,
        "awareness-02-SleepyOddOrFeverishNightmares": 0.0,
        "awareness-03-NoReactionsNoAwareness": 0.0,
        "vomit-01-No": 0.0,
        "vomit-02-Slight": 0.0,
        "vomit-03-Frequent": 0.0,
        "vomit-04-Yellow": 0.0,
        "vomit-05-5<hours": 0.0,
        "caregiverConfident": 0.0,
        "caregiverFeel": 1.0,
        "caregiverThink": 0.0
    }
}
```

A successful response should look like this:

```json
{
    "caregiver": "caution",
    "fever": "good",
    "general": "good",
    "hydration": "good",
    "medication": "caution",
    "patientState": "danger",
    "pulse": "danger",
    "respiration": "danger",
    "skin": "good"
}
```
