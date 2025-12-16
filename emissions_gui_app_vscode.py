"""
emissions_gui_app_vscode.py

Single-file Streamlit app for VS Code

Run in VS Code terminal:
1) (optional) Create + activate venv:
   python -m venv .venv
   .venv\Scripts\activate

2) Install requirements:
   pip install -r requirements.txt
   (or) pip install streamlit scikit-learn pandas matplotlib joblib

3) Run:
   streamlit run emissions_gui_app_vscode.py
"""

import os
import io
import logging
from typing import Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import joblib

# ---------- Configuration ----------
DEFAULT_DATA_FILE = "emissions_reduction_data.csv"  # relative to the script
MODEL_SAVE_PATH = "model.joblib"
RANDOM_STATE = 42
# -----------------------------------

# Logging for easier debugging in VS Code terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Emissions Reduction", layout="wide")


@st.cache_data
def load_csv_from_bytes(data_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data_bytes))


def try_load_default_csv() -> Optional[pd.DataFrame]:
    script_dir = os.path.abspath(os.path.dirname(__file__))
    candidate = os.path.join(script_dir, DEFAULT_DATA_FILE)
    if os.path.exists(candidate):
        try:
            df = pd.read_csv(candidate)
            logger.info(f"Loaded default CSV from {candidate}")
            return df
        except Exception as e:
            logger.exception("Failed to read default CSV file.")
            return None
    logger.info(f"No default CSV at {candidate}")
    return None


def prepare_features(df: pd.DataFrame, target_col: str = "emission_reduction") -> Tuple[pd.DataFrame, pd.Series, list]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe. Columns: {df.columns.tolist()}")
    # use all other numeric columns as features
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    # keep only numeric columns (or try to convert)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in X.columns if c not in numeric_cols]
    if non_numeric:
        # try to coerce non-numeric to numeric if possible
        for c in non_numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    return X, y, numeric_cols


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=int(random_state)))
    ])
    return pipe


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    evals = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, digits=4, output_dict=False),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    # try ROC AUC if probability available and binary
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            evals["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        pass
    return evals


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation="nearest")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm_arr):
        ax.text(j, i, int(val), ha="center", va="center", color="white" if val > cm_arr.max() / 2 else "black")
    fig.tight_layout()
    return fig


def main():
    st.title("Emissions Reduction")
    st.markdown("""
    
    """)

    # Data loading options
    st.sidebar.header("Data source")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    df = None
    if uploaded_file is not None:
        try:
            df = load_csv_from_bytes(uploaded_file.read())
            st.sidebar.success("Loaded uploaded CSV")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded CSV: {e}")
            logger.exception("Failed to load uploaded CSV")
    else:
        df = try_load_default_csv()
        if df is not None:
            st.sidebar.success(f"Loaded default CSV ({DEFAULT_DATA_FILE})")
        else:
            st.sidebar.info("No CSV loaded yet. Upload one or place a CSV file named "
                             f"'{DEFAULT_DATA_FILE}' next to this script and reload.")

    if df is None:
        st.warning("Please upload a CSV file or add the default CSV file and refresh.")
        st.stop()

    st.write("### Dataset preview")
    st.dataframe(df.head(20))

    # target selection
    default_target = "emission_reduction" if "emission_reduction" in df.columns else df.columns[-1]
    target_col = st.sidebar.text_input("Target column name", value=default_target)

    try:
        X, y, feature_cols = prepare_features(df, target_col=target_col)
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        logger.exception("Error preparing features")
        st.stop()

    st.sidebar.write(f"Using feature columns: {feature_cols}")
    st.sidebar.write(f"Target: {target_col}")

    # Split parameters
    test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, 0.2, step=0.05)
    random_state = st.sidebar.number_input("Random seed", value=RANDOM_STATE, step=1)

    # Load existing model if present
    model_exists = os.path.exists(MODEL_SAVE_PATH)
    if model_exists:
        st.sidebar.write(f"Saved model found: {MODEL_SAVE_PATH}")
        if st.sidebar.button("Load saved model"):
            try:
                model = joblib.load(MODEL_SAVE_PATH)
                st.sidebar.success("Model loaded from disk.")
            except Exception as e:
                st.sidebar.error(f"Failed to load model: {e}")
                logger.exception("Failed to load model")
                model = None
        else:
            model = None
    else:
        model = None

    # Train model
    if model is None:
        if st.sidebar.button("Train model now"):
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state), stratify=y if len(np.unique(y)) > 1 else None)
                pipeline = build_pipeline(random_state=int(random_state))
                pipeline.fit(X_train, y_train)
                # Save model
                joblib.dump(pipeline, MODEL_SAVE_PATH)
                st.sidebar.success(f"Model trained and saved to {MODEL_SAVE_PATH}")
                model = pipeline
                # Evaluate and show results
                evals = evaluate_model(model, X_test, y_test)
                st.subheader("Model evaluation")
                st.write(f"Accuracy: **{evals.get('accuracy', 'N/A'):.4f}**")
                if "roc_auc" in evals:
                    st.write(f"ROC AUC: **{evals['roc_auc']:.4f}**")
                st.text(evals.get("classification_report", "No report"))
                cm = evals.get("confusion_matrix", [[0,0],[0,0]])
                st.pyplot(plot_confusion_matrix(cm))
            except Exception as e:
                st.error(f"Training failed: {e}")
                logger.exception("Training failed")
                st.stop()
        else:
            st.info("Click 'Train model now' in the sidebar to train a model.")
            st.stop()

    # From here we have a trained model
    st.header("Make predictions")
    input_mode = st.selectbox("Choose input method", options=[
        "Manual single input",
        "Pick a row from dataset",
        "Upload CSV for batch prediction",
        "Randomly generate rows"
    ])

    def predict_and_display(input_df: pd.DataFrame):
        # Ensure columns match the training features
        missing = [c for c in feature_cols if c not in input_df.columns]
        if missing:
            st.error(f"Input is missing required feature columns: {missing}")
            return
        X_in = input_df[feature_cols].copy()
        preds = model.predict(X_in)
        output = input_df.reset_index(drop=True).copy()
        output["predicted_" + target_col] = preds
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_in)
                if probs.shape[1] > 1:
                    output["probability_of_1"] = probs[:, 1]
            except Exception:
                pass
        st.dataframe(output)
        csv = output.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions (CSV)", csv, file_name="predictions.csv", mime="text/csv")

    if input_mode == "Manual single input":
        st.subheader("Manual input")
        manual = {}
        # show sliders/number inputs based on training data ranges
        for c in feature_cols:
            col_min = float(X[c].min())
            col_max = float(X[c].max())
            col_mean = float(X[c].mean())
            # if range is tiny, use number_input
            if (col_max - col_min) < 1e-6:
                val = st.number_input(f"{c}", value=col_mean)
            else:
                val = st.slider(f"{c}", min_value=col_min, max_value=col_max, value=col_mean)
            manual[c] = float(val)
        if st.button("Predict (manual)"):
            input_df = pd.DataFrame([manual])
            predict_and_display(input_df)

    elif input_mode == "Pick a row from dataset":
        st.subheader("Pick a row")
        idx = st.number_input("Row index", min_value=0, max_value=len(df)-1, value=0)
        row = df.iloc[[idx]][feature_cols]
        st.write(row)
        if st.button("Predict this row"):
            predict_and_display(row)

    elif input_mode == "Upload CSV for batch prediction":
        st.subheader("Upload CSV")
        uploaded_batch = st.file_uploader("CSV with feature columns", type=["csv"], key="batch")
        if uploaded_batch is not None:
            try:
                batch_df = pd.read_csv(uploaded_batch)
                st.write("Preview of uploaded batch:")
                st.dataframe(batch_df.head(20))
                if st.button("Predict batch"):
                    predict_and_display(batch_df)
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
                logger.exception("Failed reading batch CSV")

    elif input_mode == "Randomly generate rows":
        st.subheader("Random generator")
        n = st.number_input("Number of rows", min_value=1, max_value=500, value=5)
        if st.button("Generate & predict"):
            rand_df = pd.DataFrame({
                c: np.random.uniform(X[c].min(), X[c].max(), size=int(n))
                for c in feature_cols
            })
            predict_and_display(rand_df)

    st.markdown("---")
    
if __name__ == "__main__":
    main()
