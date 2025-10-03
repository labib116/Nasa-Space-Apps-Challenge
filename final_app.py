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

# --- CRITICAL IMPORTS FOR JOBLIB LOADING (Kepler KOI Model) ---
# These are necessary for the joblib file to reconstruct the custom model class
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

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ==============================================================================
# 1. KEPLER KOI MODEL CLASS (Must be an exact copy from training script)
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
# 2. CACHED LOADING AND PREDICTION FUNCTIONS FOR BOTH PIPELINES
# ==============================================================================

# --- TSFRESH PIPELINE ---
@st.cache_resource
def load_tsfresh_resources():
    """Loads artifacts for the tsfresh model."""
    artifacts = {}
    model_files = [
        'meta_model.joblib', 'base_model_xgb_fold0.joblib', 'base_model_lgb_fold0.joblib',
        'base_model_cat_fold0.joblib', 'scaler.joblib', 'label_encoder.joblib', 'feature_names.joblib'
    ]
    if not all(os.path.exists(f) for f in model_files):
        st.error("Missing one or more required .joblib files for the tsfresh pipeline. Please check the directory.")
        return None
    for file in model_files:
        key = file.split('.')[0]
        artifacts[key] = joblib.load(file)

    if not os.path.exists('tsfresh_features.csv'):
        st.error("`tsfresh_features.csv` not found. Please place it in the same directory.")
        return None
    artifacts['data'] = pd.read_csv('tsfresh_features.csv')
    return artifacts

def run_tsfresh_prediction(artifacts, planet_id):
    """Runs prediction for the tsfresh pipeline."""
    df = artifacts['data']
    single_planet_df = df[df['pl_name'] == planet_id]
    if single_planet_df.empty:
        st.error(f"Planet with 'pl_name' '{planet_id}' not found.")
        return None

    feature_names = artifacts['feature_names']
    X_new = single_planet_df[feature_names]
    X_new_scaled = artifacts['scaler'].transform(X_new)

    xgb_preds = artifacts['base_model_xgb_fold0'].predict_proba(X_new_scaled)
    lgb_preds = artifacts['base_model_lgb_fold0'].predict_proba(X_new_scaled)
    cat_preds = artifacts['base_model_cat_fold0'].predict_proba(X_new_scaled)

    meta_features = np.hstack([xgb_preds, lgb_preds, cat_preds])
    final_preds_encoded = artifacts['meta_model'].predict(meta_features)
    final_preds_decoded = artifacts['label_encoder'].inverse_transform(final_preds_encoded)
    probabilities = artifacts['meta_model'].predict_proba(meta_features)

    results_df = single_planet_df.copy()
    results_df['Predicted Disposition'] = final_preds_decoded
    class_names = artifacts['label_encoder'].classes_
    for i, class_name in enumerate(class_names):
        results_df[f'Prob_{class_name}'] = probabilities[:, i]
    return results_df

# --- KEPLER KOI PIPELINE ---
@st.cache_resource
def load_koi_resources():
    """Loads artifacts for the Kepler KOI model."""
    resources = {}
    if not os.path.exists('ultimate_exoplanet_model.joblib'):
        st.error("`ultimate_exoplanet_model.joblib` not found.")
        return None
    resources['model'] = UltimateExoplanetStackingModel.load('ultimate_exoplanet_model.joblib')

    if not os.path.exists('kepler_koi.csv'):
        st.error("`kepler_koi.csv` not found.")
        return None
    df = pd.read_csv('kepler_koi.csv', comment='#')
    resources['data'] = df.rename(columns=lambda x: x.strip())
    return resources

def run_koi_prediction(resources, koi_ids):
    """Runs prediction for the Kepler KOI pipeline."""
    model = resources['model']
    full_df = resources['data']
    id_col = 'kepoi_name'

    koi_ids_lower = [str(i).lower() for i in koi_ids]
    df_lookup = full_df.copy()
    df_lookup[id_col] = df_lookup[id_col].str.lower()
    input_data = full_df[df_lookup[id_col].isin(koi_ids_lower)]

    if input_data.empty:
        st.warning(f"Could not find data for any of the provided KOI IDs.")
        return None

    X_predict = input_data[model.feature_names_in_]
    predictions = model.predict(X_predict)
    probabilities = model.predict_proba(X_predict)

    results_df = input_data[[id_col, 'koi_disposition']].copy()
    results_df.rename(columns={'koi_disposition': 'Actual Disposition'}, inplace=True)
    results_df['Predicted Disposition'] = predictions

    proba_df = pd.DataFrame(probabilities, columns=[f"P({c})" for c in model.classes_], index=input_data.index)
    results_df = pd.concat([results_df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    return results_df


# ==============================================================================
# 3. STREAMLIT UI AND MAIN LOGIC
# ==============================================================================
def main():
    st.set_page_config(page_title="Exoplanet Predictor Hub", layout="wide")
    st.title("🌌 Exoplanet Predictor Hub")
    st.markdown("Select a prediction pipeline below based on your dataset.")

    pipeline_choice = st.radio(
        "**Select Prediction Pipeline:**",
        ('TESS/K2 Time-Series (`tsfresh`) Model', 'Kepler KOI Feature Model'),
        horizontal=True,
    )

    st.markdown("---")

    # --- TSFRESH PIPELINE UI ---
    if pipeline_choice == 'TESS/K2 Time-Series (`tsfresh`) Model':
        st.header("🔭 Predict from `tsfresh` Time-Series Features")
        st.info("This pipeline uses a model trained on hundreds of statistical features extracted from light curves. It requires `tsfresh_features.csv` and its associated model files.")
        
        tsfresh_artifacts = load_tsfresh_resources()
        if tsfresh_artifacts:
            planet_id = st.text_input("Enter the Planet Name (`pl_name`):", placeholder="e.g., EPIC 201126583.01")
            if st.button("Predict Disposition (tsfresh)", type="primary"):
                if planet_id:
                    results = run_tsfresh_prediction(tsfresh_artifacts, planet_id.strip())
                    if results is not None:
                        st.subheader(f"Prediction Result for {planet_id.strip()}")
                        key_info_cols = ['pl_name', 'hostname', 'disposition', 'discoverymethod']
                        display_cols = [col for col in key_info_cols if col in results.columns]
                        prob_cols = [col for col in results.columns if col.startswith('Prob_')]
                        st.dataframe(results[display_cols + ['Predicted Disposition'] + prob_cols])
                else:
                    st.warning("Please enter a `pl_name`.")

    # --- KEPLER KOI PIPELINE UI ---
    elif pipeline_choice == 'Kepler KOI Feature Model':
        st.header("🛰️ Predict from Kepler KOI Features")
        st.info("This pipeline uses a model trained on the curated features from the Kepler `kepler_koi.csv` dataset. It requires that file and `ultimate_exoplanet_model.joblib`.")
        
        koi_resources = load_koi_resources()
        if koi_resources:
            id_input = st.text_area(
                "List of KOI Names (one per line):",
                value="K00752.01\nK00752.02",
                height=120
            )
            koi_ids_list = [id.strip() for id in id_input.split('\n') if id.strip()]
            if st.button("Predict Disposition (KOI)", type="primary") and koi_ids_list:
                results = run_koi_prediction(koi_resources, koi_ids_list)
                if results is not None:
                    st.subheader("Prediction Results")
                    st.dataframe(results)

                    pred_counts = results['Predicted Disposition'].value_counts().reset_index()
                    fig, ax = plt.subplots()
                    sns.barplot(x='Predicted Disposition', y='count', data=pred_counts, ax=ax, palette='viridis')
                    ax.set_title("Prediction Counts")
                    st.pyplot(fig)


if __name__ == "__main__":
    main()

