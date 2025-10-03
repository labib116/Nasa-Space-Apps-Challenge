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
# 2. CACHED LOADING FUNCTIONS
# ==============================================================================

@st.cache_resource
def load_tsfresh_resources():
    artifacts = {}
    model_files = [
        'meta_model.joblib', 'base_model_xgb_fold0.joblib', 'base_model_lgb_fold0.joblib',
        'base_model_cat_fold0.joblib', 'scaler.joblib', 'label_encoder.joblib', 'feature_names.joblib'
    ]
    if not all(os.path.exists(f) for f in model_files):
        st.sidebar.error("Missing tsfresh model files.")
        return None
    for file in model_files:
        key = file.split('.')[0]
        artifacts[key] = joblib.load(file)
    if not os.path.exists('tsfresh_features.csv'):
        st.sidebar.error("`tsfresh_features.csv` not found.")
        return None
    artifacts['data'] = pd.read_csv('tsfresh_features.csv')
    return artifacts

@st.cache_resource
def load_koi_resources():
    resources = {}
    if not os.path.exists('ultimate_exoplanet_model.joblib'):
        st.sidebar.error("`ultimate_exoplanet_model.joblib` not found.")
        return None
    resources['model'] = UltimateExoplanetStackingModel.load('ultimate_exoplanet_model.joblib')
    if not os.path.exists('kepler_koi.csv'):
        st.sidebar.error("`kepler_koi.csv` not found.")
        return None
    df = pd.read_csv('kepler_koi.csv', comment='#')
    resources['data'] = df.rename(columns=lambda x: x.strip())
    return resources

# ==============================================================================
# 3. PREDICTION HUB LOGIC
# ==============================================================================
def display_prediction_hub():
    st.header("🎯 Prediction Hub")
    st.markdown("Select a prediction pipeline below based on your dataset.")
    
    pipeline_choice = st.radio(
        "**Select Prediction Pipeline:**",
        ('TESS/K2 Time-Series (`tsfresh`) Model', 'Kepler KOI Feature Model'),
        horizontal=True, key="pipeline_selector"
    )
    st.markdown("---")

    if pipeline_choice == 'TESS/K2 Time-Series (`tsfresh`) Model':
        st.subheader("🔭 Predict from `tsfresh` Time-Series Features")
        tsfresh_artifacts = load_tsfresh_resources()
        if tsfresh_artifacts:
            planet_id = st.text_input("Enter the Planet Name (`pl_name`):", placeholder="e.g., EPIC 201126583.01")
            if st.button("Predict Disposition (tsfresh)", type="primary"):
                if planet_id:
                    # Prediction Logic
                    df = tsfresh_artifacts['data']
                    single_planet_df = df[df['pl_name'] == planet_id.strip()]
                    if single_planet_df.empty:
                        st.error(f"Planet '{planet_id.strip()}' not found.")
                    else:
                        feature_names = tsfresh_artifacts['feature_names']
                        X_new = single_planet_df[feature_names]
                        X_new_scaled = tsfresh_artifacts['scaler'].transform(X_new)
                        xgb_preds = tsfresh_artifacts['base_model_xgb_fold0'].predict_proba(X_new_scaled)
                        lgb_preds = tsfresh_artifacts['base_model_lgb_fold0'].predict_proba(X_new_scaled)
                        cat_preds = tsfresh_artifacts['base_model_cat_fold0'].predict_proba(X_new_scaled)
                        meta_features = np.hstack([xgb_preds, lgb_preds, cat_preds])
                        final_preds_encoded = tsfresh_artifacts['meta_model'].predict(meta_features)
                        final_preds_decoded = tsfresh_artifacts['label_encoder'].inverse_transform(final_preds_encoded)
                        probabilities = tsfresh_artifacts['meta_model'].predict_proba(meta_features)
                        
                        st.subheader(f"Prediction Result for {planet_id.strip()}")
                        results_df = single_planet_df.copy()
                        results_df['Predicted Disposition'] = final_preds_decoded
                        class_names = tsfresh_artifacts['label_encoder'].classes_
                        for i, class_name in enumerate(class_names):
                            results_df[f'Prob_{class_name}'] = probabilities[:, i]
                        
                        key_info_cols = ['pl_name', 'hostname', 'disposition', 'discoverymethod']
                        display_cols = [col for col in key_info_cols if col in results_df.columns]
                        prob_cols = [col for col in results_df.columns if col.startswith('Prob_')]
                        st.dataframe(results_df[display_cols + ['Predicted Disposition'] + prob_cols])

    elif pipeline_choice == 'Kepler KOI Feature Model':
        st.subheader("🛰️ Predict from Kepler KOI Features")
        koi_resources = load_koi_resources()
        if koi_resources:
            id_input = st.text_area("List of KOI Names (one per line):", value="K00752.01\nK00752.02", height=120)
            koi_ids_list = [id.strip() for id in id_input.split('\n') if id.strip()]
            if st.button("Predict Disposition (KOI)", type="primary") and koi_ids_list:
                # This function is not fully implemented in the provided snippet
                # but we assume it exists and works as intended.
                # results = run_koi_prediction(koi_resources, koi_ids_list)
                st.info("Prediction logic for KOI model would run here.")


# ==============================================================================
# 4. MODEL DETAILS HUB LOGIC
# ==============================================================================
def display_model_details():
    st.header("🔍 Model Details and Performance")
    tab1, tab2 = st.tabs(["TESS/K2 `tsfresh` Model", "Kepler KOI Feature Model"])

    with tab1:
        st.subheader("🔭 Stacking Model for `tsfresh` Features")
        st.markdown("""
        This model is a **stacking ensemble classifier** trained on a rich feature set generated by the `tsfresh` library. By analyzing the time-series data (light curves) of stars, `tsfresh` extracts hundreds of statistical features that describe the shape, periodicity, and fluctuations of the transit signals. This detailed feature set allows the model to make highly nuanced classifications.

        - **Architecture**: A stacking ensemble of **XGBoost**, **LightGBM**, and **CatBoost** as base models, with a **Logistic Regression** meta-model.
        - **Training**: The model was trained using 5-fold stratified cross-validation. Hyperparameters for the base models were tuned using Optuna.
        - **Key Advantage**: Excels at identifying complex patterns in light curve data that might be missed by models using only pre-computed features.
        """)

        st.markdown("---")
        st.subheader("Performance Metrics")
        
        report_data = {
            'precision': [0.94, 0.99, 0.93, 0.00],
            'recall': [0.98, 0.99, 0.78, 0.00],
            'f1-score': [0.96, 0.99, 0.85, 0.00],
            'support': [969, 575, 241, 8]
        }
        report_df = pd.DataFrame(report_data, index=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE', 'REFUTED'])
        
        col1, col2 = st.columns([1.2, 1.8])
        with col1:
            st.metric("Overall Accuracy", "95%")
            st.metric("Weighted Avg F1-Score", "0.95")
            st.dataframe(report_df)
        with col2:
            image_path = "images/tsfresh/tsfresh_confusion_matrix.jpg"
            if os.path.exists(image_path):
                st.image(image_path, caption="Meta-Model Confusion Matrix (on OOF predictions)")
            else:
                st.warning(f"`{image_path}` not found.")

        st.markdown("---")
        st.subheader("Feature Importance")
        st.markdown("Top 20 most influential features as determined by each base model. These features are statistical derivatives of the star's light curve.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            image_path = "images/tsfresh/tsfresh_feature_importance_xgb.jpg"
            if os.path.exists(image_path):
                st.image(image_path, caption="XGBoost Top Features")
            else:
                st.warning(f"`{image_path}` not found.")
        with col2:
            image_path = "images/tsfresh/tsfresh_feature_importance_lgb.jpg"
            if os.path.exists(image_path):
                st.image(image_path, caption="LightGBM Top Features")
            else:
                st.warning(f"`{image_path}` not found.")
        with col3:
            image_path = "images/tsfresh/tsfresh_feature_importance_cat.jpg"
            if os.path.exists(image_path):
                st.image(image_path, caption="CatBoost Top Features")
            else:
                st.warning(f"`{image_path}` not found.")

        st.markdown("---")
        st.subheader("Base Model Training Curves")
        image_path = "images/tsfresh/tsfresh_training_curves.jpg"
        if os.path.exists(image_path):
            st.image(image_path, caption="Validation performance (ROC AUC / LogLoss) during training for Fold 1.")
        else:
            st.warning(f"`{image_path}` not found.")

    with tab2:
        st.subheader("🛰️ UltimateExoplanetStackingModel (KOI)")
        st.markdown("""
        This model is an advanced **stacking ensemble classifier** designed for the Kepler Objects of Interest (KOI) dataset. It integrates a comprehensive preprocessing pipeline with a multi-layered modeling approach to deliver highly accurate and reliable predictions.

        - **Base Models**: A powerful quartet of **LightGBM**, **XGBoost**, **CatBoost**, and **Random Forest**.
        - **Meta-Model**: A **Calibrated Logistic Regression** model, which learns from the base models' predictions and provides well-calibrated probability scores.
        - **Feature Engineering**: Creates new, insightful features such as planet-star radius ratios, stellar density, and gravitational forces. It also uses **KMeans clustering** to group similar planetary systems.
        - **Preprocessing**: Implements a robust pipeline that handles missing values, scales numerical features, and one-hot encodes categorical data.
        - **Resampling**: Addresses class imbalance using **SMOTE** to ensure the model learns effectively from minority classes.
        - **Feature Selection**: Employs a LightGBM-based feature selector to retain only the most impactful features, improving speed and reducing noise.
        - **Threshold Optimization**: Instead of using a default 0.5 probability threshold, the model calculates the optimal threshold for each class to maximize the F1-score, leading to better practical performance.
        """)
        
        st.markdown("---")
        st.subheader("Performance Metrics on Test Set")

        # Data transcribed from the classification report screenshot
        koi_report_data = {
            'precision': [0.96, 0.99, 0.86],
            'recall': [0.99, 0.98, 0.88],
            'f1-score': [0.97, 0.98, 0.87],
            'support': [484, 287, 120]
        }
        koi_report_df = pd.DataFrame(koi_report_data, index=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])

        col1, col2 = st.columns([1.5, 1.5])
        with col1:
            st.metric("Overall Accuracy", "0.9587")
            st.metric("Weighted ROC AUC", "0.9928")
            st.dataframe(koi_report_df)
        
        with col2:
            st.markdown("**F1-Scores by Class**")
            fig, ax = plt.subplots(figsize=(6, 4))
            f1_scores = koi_report_df['f1-score']
            sns.barplot(x=f1_scores.index, y=f1_scores.values, ax=ax, palette="mako")
            ax.set_ylabel("F1-Score")
            ax.set_ylim(0, 1.05)
            ax.set_title("F1-Score per Disposition Class", fontsize=12)
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Confusion Matrix")
        image_path = "images/koi/koi_confusion_matrix.jpg"
        if os.path.exists(image_path):
            st.image(image_path, caption="Confusion Matrix on the test set, showing actual vs. predicted labels.")
        else:
            st.warning(f"Image not found at `{image_path}`. Please create the folder and add the renamed image.")
            
        st.markdown("---")
        st.subheader("Feature Importance")
        st.info("Feature importance plots for the KOI base models are not available at this time but would be displayed here.")


# ==============================================================================
# 5. MAIN APP ROUTER
# ==============================================================================
def main():
    st.set_page_config(page_title="Exoplanet Predictor Hub", layout="wide")
    st.sidebar.title("🌌 Exoplanet Predictor Hub")
    
    app_mode = st.sidebar.radio(
        "Choose a section",
        ["Prediction Hub", "Model Details"]
    )

    if app_mode == "Prediction Hub":
        display_prediction_hub()
    elif app_mode == "Model Details":
        display_model_details()

if __name__ == "__main__":
    main()

