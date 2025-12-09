import streamlit as st
import os
import json
import subprocess
import time
from pathlib import Path

INTENTS_PATH = "nlu_engine/intents.json"
MODEL_DIR = "models/intent_model"
LOG_PATH = "models/training.log"

os.makedirs("models", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================================
# PAGE CONFIG + CSS
# =====================================================================
st.set_page_config(page_title="BankBot NLU lavanya", layout="wide")

st.markdown("""
<style>
.title { font-size: 32px; font-weight: 700; color: #1a73e8; margin-bottom: 20px; }
.section-title { font-size: 22px; font-weight: 600; margin-top: 20px; }
.intent-score {
    padding: 12px; background: #eef2ff; border-left: 7px solid #4a63e7;
    margin-bottom: 10px; border-radius: 10px; font-size: 18px;
}
.entity-card {
    padding: 12px; background: #e9f7ef; border-left: 7px solid #29a846;
    margin-bottom: 10px; border-radius: 10px; display: flex; 
    gap: 10px; align-items: center;
}
.entity-icon { font-size: 24px; }
.entity-text { font-size: 16px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>BankBot NLU Engine</div>", unsafe_allow_html=True)


# =====================================================================
# INTENTS LOADING / SAVING
# =====================================================================
def load_intents():
    if not os.path.exists(INTENTS_PATH):
        return {"intents": {}}

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supported formats:
    # {
    #   "intents": { "intent_name": { "examples": [] } }
    # }
    if isinstance(data, dict):
        if isinstance(data.get("intents"), dict):
            return data
        if isinstance(data.get("intents"), list):
            return {"intents": {i["name"]: {"examples": i["examples"]} for i in data["intents"]}}

    # If file is a list
    if isinstance(data, list):
        return {"intents": {i["name"]: {"examples": i["examples"]} for i in data}}

    return {"intents": {}}


def save_intents(data):
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =====================================================================
# MODEL CHECK
# =====================================================================
def model_exists():
    required = [
        f"{MODEL_DIR}/intent_model.pkl",
        f"{MODEL_DIR}/vectorizer.pkl",
        f"{MODEL_DIR}/labels.pkl"
    ]
    return all(os.path.exists(x) for x in required)


# Load intents
intents_data = load_intents()
intents = intents_data["intents"]

# =====================================================================
# PAGE LAYOUT
# =====================================================================
col1, col2 = st.columns([1, 1.4])

# =====================================================================
# LEFT PANEL – INTENT EDITOR
# =====================================================================
with col1:
    st.markdown("<div class='section-title'>Intent List & Examples</div>", unsafe_allow_html=True)

    edited = {}
    for name, obj in intents.items():
        with st.expander(f"{name} ({len(obj['examples'])} examples)"):
            updated = st.text_area(
                f"Examples for {name}",
                "\n".join(obj["examples"]),
                key=f"txt_{name}"
            )
            edited[name] = updated

    if st.button("Save Changes"):
        for name, text in edited.items():
            intents[name]["examples"] = [x.strip() for x in text.split("\n") if x.strip()]

        save_intents(intents_data)
        st.success("Intents updated!")

    st.markdown("<hr>")
    st.markdown("<div class='section-title'>Add New Intent</div>", unsafe_allow_html=True)

    new_intent = st.text_input("Intent Name")
    new_examples = st.text_area("Examples (one per line)")

    if st.button("Add Intent"):
        if new_intent.strip() and new_examples.strip():
            intents[new_intent] = {
                "examples": [x.strip() for x in new_examples.split("\n") if x.strip()]
            }
            save_intents(intents_data)
            st.success("Intent added! Refresh page.")
        else:
            st.error("Enter intent name and at least one example.")


# =====================================================================
# RIGHT PANEL – NLU VISUALIZER
# =====================================================================
with col2:
    st.markdown("<div class='section-title'>NLU Visualizer</div>", unsafe_allow_html=True)

    query = st.text_area("Enter User Message", height=90)
    top_k = st.number_input("Top K Intents", 1, 10, 4)

    if st.button("Analyze"):
        if not model_exists():
            st.error("Model not trained yet.")
        else:
            from nlu_engine.intent_classifier import IntentClassifier
            from nlu_engine.entity_extractor import EntityExtractor

            ic = IntentClassifier()
            ex = EntityExtractor()

            intents_pred = ic.predict(query, top_k)   # → list of dicts
            entities = ex.extract(query)              # → list of (start, end, type, value)

            # ---------------------------------------
            # INTENT PREDICTION
            # ---------------------------------------
            st.markdown("<div class='section-title'>Intent Recognition</div>", unsafe_allow_html=True)

            for item in intents_pred:
                name, score = item   # ← FIXED
                st.markdown(
                           f"""
                     <div class='intent-score'>
                     <b>{name}</b> — {score:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

            # ---------------------------------------
            # ENTITY EXTRACTION
            # ---------------------------------------
            st.markdown("<div class='section-title'>Entity Extraction</div>", unsafe_allow_html=True)

            if not entities:
                st.info("No entities found.")
            else:
              for etype, val in entities:
               st.markdown(
        f"""
        <div class='entity-card'>
            <div class='entity-icon'>🔎</div>
            <div class='entity-text'>
                <b>{etype}</b><br>{val}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )




# =====================================================================
# TRAINING SECTION
# =====================================================================
st.markdown("---")
st.markdown("<div class='section-title'>Train Intent Model</div>", unsafe_allow_html=True)

epochs = st.number_input("Epochs", 1, 20, 5)
batch = st.number_input("Batch Size", 1, 32, 4)
lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=2e-5, format="%.6f")


def start_training():
    if model_exists():
        st.warning("Model already exists — using existing model.")
        return

    # Delete old partial files
    for f in ["intent_model.pkl", "vectorizer.pkl", "labels.pkl"]:
        fp = f"{MODEL_DIR}/{f}"
        if os.path.exists(fp):
            os.remove(fp)

    cmd = [
        "python",
        "nlu_engine/train_intent.py",
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--lr", str(lr)
    ]

    with open(LOG_PATH, "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f)

    st.info("Training started...")

    placeholder = st.empty()

    # Live log monitor
    while True:
        time.sleep(1)

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as f:
                placeholder.text(f.read())

        if model_exists():
            break

    st.success("🎉 Training Completed!")


if st.button("Start Training"):
    start_training()
