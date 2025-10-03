import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Exoplanet Disposition Prediction",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
    }
    .title-text {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
    }
    .header-text {
        font-size: 1.5rem;
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)


# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_models():
    """Load all model artifacts. The cache ensures this runs only once."""
    artifacts = {}
    model_files = [
        'meta_model.joblib', 'base_model_xgb_fold0.joblib', 'base_model_lgb_fold0.joblib',
        'base_model_cat_fold0.joblib', 'scaler.joblib', 'label_encoder.joblib', 'feature_names.joblib'
    ]
    missing_files = [f for f in model_files if not os.path.exists(f)]
    if missing_files:
        return None, f"Error: The following model files are missing: {', '.join(missing_files)}."

    try:
        for file in model_files:
            key = file.split('.')[0]
            artifacts[key] = joblib.load(file)
        return artifacts, None
    except Exception as e:
        return None, f"An error occurred while loading the models: {e}"

@st.cache_data
def load_data(file_path='tsfresh_features.csv'):
    """Load the local tsfresh features CSV file."""
    if not os.path.exists(file_path):
        return None, f"Error: `{file_path}` not found. Please place it in the same directory as the app."
    try:
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e:
        return None, f"An error occurred while loading `{file_path}`: {e}"


# --- PREDICTION FUNCTION ---
def predict(df, artifacts):
    """Run the full inference pipeline."""
    feature_names = artifacts['feature_names']
    cleaned_df = df.copy()

    # The cleaning steps are kept for robustness in case the CSV has issues
    if 'pl_name' in cleaned_df.columns and 'default_flag' in cleaned_df.columns:
        cleaned_df.sort_values(['pl_name', 'default_flag'], ascending=[True, False], inplace=True)
        cleaned_df.drop_duplicates(subset=['pl_name'], inplace=True, keep='first')

    missing_cols = set(feature_names) - set(cleaned_df.columns)
    if missing_cols:
        return None, None, f"Error: Data is missing required model features: {', '.join(missing_cols)}"

    X_new = cleaned_df[feature_names]
    X_new_numeric = X_new.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_new_scaled = artifacts['scaler'].transform(X_new_numeric)

    xgb_preds = artifacts['base_model_xgb_fold0'].predict_proba(X_new_scaled)
    lgb_preds = artifacts['base_model_lgb_fold0'].predict_proba(X_new_scaled)
    cat_preds = artifacts['base_model_cat_fold0'].predict_proba(X_new_scaled)

    meta_features = np.hstack([xgb_preds, lgb_preds, cat_preds])
    final_predictions_encoded = artifacts['meta_model'].predict(meta_features)
    final_predictions_decoded = artifacts['label_encoder'].inverse_transform(final_predictions_encoded)
    final_probabilities = artifacts['meta_model'].predict_proba(meta_features)
    
    return final_predictions_decoded, final_probabilities, cleaned_df


# --- STREAMLIT UI ---
st.markdown("<h1 class='title-text'>🪐 Exoplanet Disposition Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-text'>Enter a Planet Name (pl_name) to get a prediction from the locally stored data.</p>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("About the Model")
    st.write("""
    This web app uses a sophisticated **stacking ensemble model** to classify exoplanets based on pre-computed `tsfresh` features.
    - **Base Models**: XGBoost, LightGBM, CatBoost
    - **Meta-Model**: Logistic Regression
    """)
    st.header("Instructions")
    st.info("""
    1. **Place your `tsfresh_features.csv` file** in the same directory as this app.
    2. The app will load the data automatically on startup.
    3. **Enter a `pl_name`** (e.g., `EPIC 201126583.01`) into the text box.
    4. Click the **'Predict'** button to see the classification.
    """)

# --- Main Application Logic ---
artifacts, error_msg = load_models()

if error_msg:
    st.error(error_msg)
else:
    input_df, data_error_msg = load_data()

    if data_error_msg:
        st.error(data_error_msg)
    else:
        st.success("`tsfresh_features.csv` loaded successfully from the local folder.")
        
        if 'pl_name_input' not in st.session_state:
            st.session_state.pl_name_input = ""
        
        st.session_state.pl_name_input = st.text_input(
            "Enter the 'pl_name' of the planet to predict:",
            placeholder="e.g., EPIC 201126583.01",
            value=st.session_state.pl_name_input
        )

        if st.button("✨ Predict Disposition"):
            planet_id = st.session_state.pl_name_input.strip()
            if not planet_id:
                st.warning("Please enter a 'pl_name' to get a prediction.")
            else:
                single_planet_df = input_df[input_df['pl_name'] == planet_id]

                if single_planet_df.empty:
                    st.error(f"Error: Planet with 'pl_name' '{planet_id}' not found in `tsfresh_features.csv`.")
                else:
                    with st.spinner(f"Analyzing '{planet_id}'..."):
                        predictions, probabilities, cleaned_df = predict(single_planet_df, artifacts)

                        if predictions is not None:
                            results_df = cleaned_df.copy()
                            results_df['Predicted Disposition'] = predictions
                            class_names = artifacts['label_encoder'].classes_
                            for i, class_name in enumerate(class_names):
                                results_df[f'Prob_{class_name}'] = probabilities[:, i]

                            st.subheader(f"📊 Prediction Result for {planet_id}")

                            # Select key columns for a clean, informative display
                            key_info_cols = [
                                'pl_name', 'hostname', 'disposition', 'discoverymethod',
                                'pl_orbper', 'pl_rade', 'pl_masse', 'sy_dist'
                            ]
                            display_cols = [col for col in key_info_cols if col in results_df.columns]
                            prediction_cols = ['Predicted Disposition'] + [f'Prob_{c}' for c in class_names]
                            
                            st.dataframe(results_df[display_cols + prediction_cols])
                        else:
                            st.error("Prediction failed. An error occurred during the process.")

