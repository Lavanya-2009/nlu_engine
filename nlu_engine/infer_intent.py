import os
import json
import math


class IntentClassifier:
    def __init__(self, model_dir="models/intent_model"):
        """
        Loads intent label list + intent keyword map.
        If the model folder doesn't exist, a simple fallback model is used.
        """
        self.model_dir = model_dir
        self.intents = []
        self.keyword_map = {}     # {intent: [keywords]}

        if os.path.exists(model_dir):
            try:
                with open(os.path.join(model_dir, "intents.json"), "r") as f:
                    data = json.load(f)
                    self.intents = data.get("intents", [])
                    self.keyword_map = data.get("keywords", {})
            except Exception as e:
                print("Model load error:", e)
                self._load_fallback()
        else:
            # Use fallback model
            self._load_fallback()

    def _load_fallback(self):
        """Fallback simple intent model."""
        self.intents = ["transfer_money", "check_balance", "transaction_status"]

        self.keyword_map = {
            "transfer_money": ["transfer", "send", "pay", "deposit"],
            "check_balance": ["balance", "available", "funds"],
            "transaction_status": ["txn", "transaction", "status", "utr"]
        }

    def predict(self, text, top_k=1):
        text_lower = text.lower()
        scores = {}

        # simple scoring based on keyword occurrences
        for intent, words in self.keyword_map.items():
            score = sum(1 for w in words if w in text_lower)
            # avoid zero scores
            scores[intent] = score + 1e-6

        # sort intents by score high → low
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # return only top_k
        top_results = [{"intent": intent, "score": score} for intent, score in sorted_intents[:top_k]]
        return top_results


if __name__ == "__main__":
    try:
        ic = IntentClassifier()
        print(ic.predict("please transfer 5000 to my savings account", top_k=3))
    except Exception as e:
        print("Error:", e)
