🌌 Unified Exoplanet Predictor HubA Streamlit web application that serves as a centralized hub for predicting the disposition of exoplanet candidates using two distinct, powerful machine learning pipelines. Users can select the appropriate model based on their input dataset—either features extracted from time-series data (tsfresh) or curated features from the Kepler Objects of Interest (KOI) catalog.App Demo(This is a placeholder: You can replace this with a screenshot or a GIF of your running application.)✨ FeaturesDual Prediction Pipelines: Seamlessly switch between two different prediction models within a single interface.Interactive UI: A clean and user-friendly web interface built with Streamlit.Real-time Inference: Get instant predictions by providing the planet or KOI identifier.Detailed Results: View prediction outcomes, class probabilities, and relevant metadata.Data Visualization: Includes a simple plot for visualizing the distribution of predictions for the Kepler KOI pipeline.Self-Contained: The application runs locally, loading all necessary model files and datasets from the project directory.🚀 How It WorksThe application allows you to choose between two independent modeling approaches:1. TESS/K2 Time-Series (tsfresh) ModelInput Data: tsfresh_features.csvDescription: This pipeline uses a stacking model trained on over 700 statistical features extracted from the light curves of stars. It is ideal for classifying candidates based on the nuanced patterns of their transit signals.Identifier: pl_name (e.g., EPIC 201126583.01)2. Kepler KOI Feature ModelInput Data: kepler_koi.csvDescription: This pipeline uses the UltimateExoplanetStackingModel, a custom-built stacking classifier trained on the manually curated features from the official Kepler KOI dataset. It performs sophisticated feature engineering and uses a custom prediction threshold logic.Identifier: kepoi_name (e.g., K00752.01)📁 Project StructureFor the application to run correctly, your project folder should have the following structure:.
├── 📄 app.py
├── 📄 kepler_koi.csv
├── 📄 tsfresh_features.csv
├── 📄 ultimate_exoplanet_model.joblib
├── 📄 meta_model.joblib
├── 📄 base_model_xgb_fold0.joblib
├── 📄 base_model_lgb_fold0.joblib
├── 📄 base_model_cat_fold0.joblib
├── 📄 scaler.joblib
├── 📄 label_encoder.joblib
├── 📄 feature_names.joblib
├── 📄 requirements.txt
└── 📄 README.md
🛠️ Setup and InstallationClone the Repository / Download Files:Ensure all the required files listed above are in a single project folder.Create a Virtual Environment (Recommended):python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install Dependencies:A requirements.txt file is provided for easy installation.pip install -r requirements.txt
Run the Streamlit App:Open your terminal, navigate to the project directory, and run the following command:streamlit run app.py
Your web browser should automatically open with the application running.📖 UsageSelect a Pipeline:At the top of the page, choose either the 'TESS/K2 Time-Series (tsfresh) Model' or the 'Kepler KOI Feature Model'.Enter Identifiers:If you chose the tsfresh model, enter a single pl_name into the text input field.If you chose the Kepler KOI model, enter one or more kepoi_name identifiers into the text area, with each ID on a new line.Get Predictions:Click the "Predict Disposition" button. The application will load the corresponding data, run the inference pipeline, and display the results in a table.LicenseThis project is licensed under the MIT License. See the LICENSE file for details.
