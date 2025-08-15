import joblib, pandas as pd

class InferenceEngine:
    def __init__(self, model_pkl, scaler_pkl):
        self.model = joblib.load(model_pkl)
        self.scaler = joblib.load(scaler_pkl)
        self.feature_cols = self.model.feature_names_in_

    def predict(self, feature_row: dict):
        X = pd.DataFrame([feature_row])[self.feature_cols]
        Xs = self.scaler.transform(X)
        proba = self.model.predict_proba(Xs)[0]
        classes = list(self.model.classes_)
        idx = proba.argmax()
        return classes[idx], float(proba[idx])
