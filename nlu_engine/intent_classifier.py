import joblib
import numpy as np

class IntentClassifier:
    def __init__(self, model_path="models/intent_model"):
        self.model = joblib.load(f"{model_path}/intent_model.pkl")
        self.vectorizer = joblib.load(f"{model_path}/vectorizer.pkl")
        self.labels = joblib.load(f"{model_path}/labels.pkl")

        # reverse lookup
        self.id2label = {v: k for k, v in self.labels.items()}

    def predict(self, text, top_k=3):
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]

        top_ids = np.argsort(probs)[::-1][:top_k]

        # RETURN tuple (intent, confidence)
        results = [
            (self.id2label[i], float(probs[i]))
            for i in top_ids
        ]

        return results
