import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# --- CRITICAL IMPORTS FOR JOBLIB LOADING ---
# These imports must be present for joblib to reconstruct the model object
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
# -------------------------------------------

# Suppress warnings and fix OpenBLAS conflict for stability
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ==============================================================================
# 1. ULTIMATE EXOPLANET MODEL CLASS
# This class MUST be an exact copy of the one from your final training script.
# ==============================================================================

class UltimateExoplanetStackingModel:
    """Model class definition, replicated exactly from the training script."""
    def __init__(self, base_models, meta_model, n_splits=5, random_state=42, 
                 resampling_strategy='smote', use_feature_selection=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.resampling_strategy = resampling_strategy
        self.use_feature_selection = use_feature_selection
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.feature_selector_ = None
        self.fitted_base_models_ = {}
        self.fitted_calibrator_ = None
        self.optimal_thresholds_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.classes_ = None
        self.kmeans_ = None

    def _feature_engineer(self, X, is_training=False):
        """Applies feature engineering steps."""
        X_eng = X.copy()
        epsilon = 1e-6
        X_eng['planet_star_radius_ratio'] = X_eng['koi_prad'] / (X_eng['koi_srad'] + epsilon)
        X_eng['stellar_gravity'] = X_eng['koi_smass'] / (X_eng['koi_srad']**2 + epsilon)
        X_eng['stellar_density'] = X_eng['koi_smass'] / (X_eng['koi_srad']**3 + epsilon)
        X_eng['duration_over_period'] = X_eng['koi_duration'] / (X_eng['koi_period'] + epsilon)
        X_eng['depth_to_snr'] = X_eng['koi_depth'] / (X_eng['koi_model_snr'] + epsilon)
        temp_bins = [0, 3000, 5000, 6000, 7500, np.inf]
        temp_labels = ['M-type', 'K-type', 'G-type', 'F-type', 'A-type']
        X_eng['star_type'] = pd.cut(X_eng['koi_steff'], bins=temp_bins, labels=temp_labels)
        cluster_features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
        X_cluster_imputed = X_eng[cluster_features].fillna(X_eng[cluster_features].median())
        
        if is_training:
            self.kmeans_ = KMeans(n_clusters=5, random_state=self.random_state, n_init='auto')
            X_eng['koi_cluster'] = self.kmeans_.fit_predict(X_cluster_imputed)
        else:
            if not hasattr(self, 'kmeans_') or self.kmeans_ is None:
                raise RuntimeError("KMeans model has not been fitted. Call train first.")
            try:
                check_is_fitted(self.kmeans_)
                X_eng['koi_cluster'] = self.kmeans_.predict(X_cluster_imputed)
            except NotFittedError:
                 raise RuntimeError("KMeans model is not fitted. Loaded model might be corrupted.")
        return X_eng

    def predict_proba(self, X_new):
        """Generates class probabilities."""
        if not self.fitted_calibrator_: 
            raise RuntimeError("Model has not been trained or loaded correctly.")
        X_new_eng = self._feature_engineer(X_new.loc[:, self.feature_names_in_], is_training=False)
        X_new_proc = self.preprocessor.transform(X_new_eng)
        X_new_final = self.feature_selector_.transform(X_new_proc) if self.use_feature_selection else X_new_proc
        base_predictions = [model.predict_proba(X_new_final) for model in self.fitted_base_models_.values()]
        stacked_base_predictions = np.hstack(base_predictions)
        return self.fitted_calibrator_.predict_proba(stacked_base_predictions)

    def predict(self, X_new):
        """Generates final class predictions."""
        probabilities = self.predict_proba(X_new)
        preds = np.zeros(len(probabilities), dtype=int)
        optimal_thresholds_list = [self.optimal_thresholds_[i] for i in range(len(self.classes_))]
        for i in range(len(probabilities)):
            passed_classes = np.where(probabilities[i] >= optimal_thresholds_list)[0]
            if len(passed_classes) == 0:
                preds[i] = np.argmax(probabilities[i])
            else:
                preds[i] = passed_classes[np.argmax(probabilities[i][passed_classes])]
        return self.label_encoder.inverse_transform(preds)

    @staticmethod
    def load(filepath):
        """Loads the model using joblib."""
        return joblib.load(filepath)

# ==============================================================================
# 2. STREAMLIT APPLICATION LOGIC
# ==============================================================================

MODEL_FILE = 'ultimate_exoplanet_model.joblib'
DATA_FILE = 'kepler_koi.csv'

@st.cache_resource
def load_resources():
    """Loads the model and the full dataset."""
    st.write("---")
    try:
        model = UltimateExoplanetStackingModel.load(MODEL_FILE)
        st.success("✅ **Model loaded successfully!**")
    except Exception as e:
        st.error(f"❌ **Error loading model file '{MODEL_FILE}'**.")
        st.info("Ensure the model file is in the same folder and you have installed the correct libraries from `requirements.txt`.")
        st.exception(e)
        st.stop()
        
    try:
        full_df = pd.read_csv(DATA_FILE, comment='#')
        # This mirrors the training script: strip whitespace but keep original case.
        full_df = full_df.rename(columns=lambda x: x.strip())
        st.success(f"✅ **Feature Data ('{DATA_FILE}') loaded successfully** ({len(full_df)} rows).")
    except FileNotFoundError:
        st.error(f"❌ **Error: Cannot find data file '{DATA_FILE}'**.")
        st.info("Please make sure the CSV file is in the exact same folder as your Streamlit app script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ **An error occurred while loading '{DATA_FILE}'**.")
        st.exception(e)
        st.stop()
        
    return model, full_df

def run_inference(model, full_df, koi_ids):
    """Retrieves data, runs inference, and returns results."""
    id_col = 'kepoi_name'
    
    # Standardize input IDs and the lookup column for robust, case-insensitive matching
    koi_ids_lower = [str(i).lower() for i in koi_ids]
    df_lookup = full_df.copy()
    df_lookup[id_col] = df_lookup[id_col].str.lower()

    input_data = full_df[df_lookup[id_col].isin(koi_ids_lower)]
    
    if input_data.empty:
        st.warning(f"⚠️ **Could not find data** for any of the provided IDs: {', '.join(koi_ids)}.")
        return None, None

    # This works because the columns in `input_data` have their original casing,
    # matching what `model.feature_names_in_` expects from the new training.
    X_predict = input_data[model.feature_names_in_]
    
    try:
        with st.spinner("🚀 Running prediction pipeline..."):
            predictions = model.predict(X_predict)
            probabilities = model.predict_proba(X_predict)
            
        results_df = input_data[[id_col, 'koi_disposition']].copy()
        results_df.rename(columns={'koi_disposition': 'Actual Disposition'}, inplace=True)
        results_df['Predicted Disposition'] = predictions
        
        proba_df = pd.DataFrame(probabilities, columns=[f"P({c})" for c in model.classes_], index=input_data.index)
        results_df = pd.concat([results_df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
        
        return results_df
        
    except Exception as e:
        st.error("An error occurred during the prediction step.")
        st.exception(e)
        return None

def main():
    """Main Streamlit App Function."""
    st.set_page_config(page_title="Exoplanet Predictor", layout="wide")

    st.title("🌌 Ultimate Exoplanet Stacking Model Predictor")
    st.markdown(f"This app uses a pre-trained model to predict the disposition of Kepler Objects of Interest (KOIs) based on data from **`{DATA_FILE}`**.")
    
    model, full_df = load_resources()

    st.subheader("1. Enter KOI Names for Prediction")
    st.info(f"Enter one or more KOI names (e.g., **K00752.01**) to retrieve their features and run a prediction.")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        id_input = st.text_area(
            "List of KOI Names (one per line):",
            value="K00752.01\nK00752.02\nK00115.01",
            height=150
        )
        
    with col2:
        st.markdown("##### Model Configuration")
        st.metric("Input Features Expected", len(model.feature_names_in_))
        st.metric("Model Classes", ", ".join(model.classes_))
        st.markdown("---")
        if hasattr(model, 'optimal_thresholds_') and model.optimal_thresholds_:
            st.caption(f"Optimal Thresholds: { {int(k): f'{v:.3f}' for k, v in model.optimal_thresholds_.items()} }")
        
    koi_ids_list = [id.strip() for id in id_input.split('\n') if id.strip()]

    if st.button("Predict Disposition", type="primary") and koi_ids_list:
        results_df = run_inference(model, full_df, koi_ids_list)
        
        if results_df is not None:
            st.subheader("2. Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            if 'Predicted Disposition' in results_df.columns:
                st.subheader("3. Visualization")
                
                pred_counts = results_df['Predicted Disposition'].value_counts().reset_index()
                pred_counts.columns = ['Disposition', 'Count']
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x='Disposition', y='Count', data=pred_counts, palette="viridis", ax=ax)
                ax.set_title("Predicted Disposition Counts")
                ax.set_xlabel("Predicted Class")
                ax.set_ylabel("Number of KOIs")
                st.pyplot(fig)
                plt.close(fig)

if __name__ == "__main__":
    main()

