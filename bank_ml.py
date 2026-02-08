import joblib
import pandas as pd

class BankModel:
    def __init__(self):
        self.model = joblib.load('./model/model.pkl')

    def predict_loan(self, job, age, campaign, euribor3m, threshold=0.25):
        data = pd.DataFrame([{
            'job': job,
            'age': age,
            'campaign': campaign,
            'euribor3m': euribor3m
        }])

        prob = self.model.predict_proba(data)[0][1]
        pred = 1 if prob >= threshold else 0

        return {
            "prediction": pred,
            "probability": prob
        }