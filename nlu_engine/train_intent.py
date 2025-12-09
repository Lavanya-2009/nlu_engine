import argparse
import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load_intents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []

    # ----------------------------
    # FORMAT 1: {"intents": { "name": {examples: []}}}
    # ----------------------------
    if isinstance(data, dict) and isinstance(data.get("intents"), dict):
        for intent_name, obj in data["intents"].items():
            for ex in obj["examples"]:
                texts.append(ex)
                labels.append(intent_name)
        return texts, labels

    # ----------------------------
    # FORMAT 2: {"intents": [ {name: "", examples: []} ]}
    # ----------------------------
    if isinstance(data, dict) and isinstance(data.get("intents"), list):
        for it in data["intents"]:
            intent_name = it["name"]
            for ex in it["examples"]:
                texts.append(ex)
                labels.append(intent_name)
        return texts, labels

    # ----------------------------
    # FORMAT 3: [ {name: "", examples: []} ]
    # ----------------------------
    if isinstance(data, list):
        for it in data:
            intent_name = it["name"]
            for ex in it["examples"]:
                texts.append(ex)
                labels.append(intent_name)
        return texts, labels

    raise ValueError("❌ Invalid intents.json format!")


def train_model(intents_file, out_dir, epochs, batch, lr):
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print("Model already exists. Skipping training.")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("Training started...")
    print(f"Intents file: {intents_file}")
    print(f"Output dir: {out_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Learning rate: {lr}")

    texts, labels = load_intents(intents_file)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    label2id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    y = [label2id[label] for label in labels]

    clf = LogisticRegression(max_iter=epochs)
    clf.fit(X, y)

    joblib.dump(clf, f"{out_dir}/intent_model.pkl")
    joblib.dump(vectorizer, f"{out_dir}/vectorizer.pkl")
    joblib.dump(label2id, f"{out_dir}/labels.pkl")

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--intents", type=str, default="nlu_engine/intents.json")
    parser.add_argument("--out_dir", type=str, default="models/intent_model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()

    train_model(
        intents_file=args.intents,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr
    )
