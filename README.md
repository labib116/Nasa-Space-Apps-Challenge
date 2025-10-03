<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Exoplanet Predictor Hub</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        h1 { font-size: 2em; }
        h2 { font-size: 1.5em; }
        h3 { font-size: 1.25em; }
        code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            background-color: rgba(27,31,35,0.05);
            padding: 0.2em 0.4em;
            margin: 0;
            font-size: 85%;
            border-radius: 3px;
        }
        pre {
            background-color: #f6f8fa;
            border-radius: 3px;
            font-size: 85%;
            line-height: 1.45;
            overflow: auto;
            padding: 16px;
        }
        pre code {
            background-color: transparent;
            padding: 0;
            margin: 0;
            font-size: 100%;
        }
        ul, ol {
            padding-left: 2em;
        }
        li {
            margin-bottom: 0.5em;
        }
        strong {
            font-weight: 600;
        }
    </style>
</head>
<body>

    <h1>🌌 Unified Exoplanet Predictor Hub</h1>

    <p>A Streamlit web application that serves as a centralized hub for predicting the disposition of exoplanet candidates using two distinct, powerful machine learning pipelines. Users can select the appropriate model based on their input dataset—either features extracted from time-series data (<code>tsfresh</code>) or curated features from the Kepler Objects of Interest (KOI) catalog.</p>

    <h3>App Demo</h3>
    <p><em>(This is a placeholder: You can replace this with a screenshot or a GIF of your running application.)</em></p>

    <h2>✨ Features</h2>
    <ul>
        <li><strong>Dual Prediction Pipelines</strong>: Seamlessly switch between two different prediction models within a single interface.</li>
        <li><strong>Interactive UI</strong>: A clean and user-friendly web interface built with Streamlit.</li>
        <li><strong>Real-time Inference</strong>: Get instant predictions by providing the planet or KOI identifier.</li>
        <li><strong>Detailed Results</strong>: View prediction outcomes, class probabilities, and relevant metadata.</li>
        <li><strong>Data Visualization</strong>: Includes a simple plot for visualizing the distribution of predictions for the Kepler KOI pipeline.</li>
        <li><strong>Self-Contained</strong>: The application runs locally, loading all necessary model files and datasets from the project directory.</li>
    </ul>

    <h2>🚀 How It Works</h2>
    <p>The application allows you to choose between two independent modeling approaches:</p>

    <h3>1. TESS/K2 Time-Series (<code>tsfresh</code>) Model</h3>
    <ul>
        <li><strong>Input Data</strong>: <code>tsfresh_features.csv</code></li>
        <li><strong>Description</strong>: This pipeline uses a stacking model trained on over 700 statistical features extracted from the light curves of stars. It is ideal for classifying candidates based on the nuanced patterns of their transit signals.</li>
        <li><strong>Identifier</strong>: <code>pl_name</code> (e.g., <code>EPIC 201126583.01</code>)</li>
    </ul>

    <h3>2. Kepler KOI Feature Model</h3>
    <ul>
        <li><strong>Input Data</strong>: <code>kepler_koi.csv</code></li>
        <li><strong>Description</strong>: This pipeline uses the <code>UltimateExoplanetStackingModel</code>, a custom-built stacking classifier trained on the manually curated features from the official Kepler KOI dataset. It performs sophisticated feature engineering and uses a custom prediction threshold logic.</li>
        <li><strong>Identifier</strong>: <code>kepoi_name</code> (e.g., <code>K00752.01</code>)</li>
    </ul>

    <h2>📁 Project Structure</h2>
    <p>For the application to run correctly, your project folder should have the following structure:</p>
    <pre><code>.
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
</code></pre>

    <h2>🛠️ Setup and Installation</h2>
    <ol>
        <li><strong>Clone the Repository / Download Files:</strong><br>
        Ensure all the required files listed above are in a single project folder.</li>
        <li><strong>Create a Virtual Environment (Recommended):</strong>
        <pre><code>python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`</code></pre>
        </li>
        <li><strong>Install Dependencies:</strong><br>
        A <code>requirements.txt</code> file is provided for easy installation.
        <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li><strong>Run the Streamlit App:</strong><br>
        Open your terminal, navigate to the project directory, and run the following command:
        <pre><code>streamlit run app.py</code></pre>
        Your web browser should automatically open with the application running.</li>
    </ol>

    <h2>📖 Usage</h2>
    <ol>
        <li><strong>Select a Pipeline:</strong><br>
        At the top of the page, choose either the <strong>'TESS/K2 Time-Series (<code>tsfresh</code>) Model'</strong> or the <strong>'Kepler KOI Feature Model'</strong>.</li>
        <li><strong>Enter Identifiers:</strong>
        <ul>
            <li>If you chose the <code>tsfresh</code> model, enter a single <code>pl_name</code> into the text input field.</li>
            <li>If you chose the Kepler KOI model, enter one or more <code>kepoi_name</code> identifiers into the text area, with each ID on a new line.</li>
        </ul>
        </li>
        <li><strong>Get Predictions:</strong><br>
        Click the <strong>"Predict Disposition"</strong> button. The application will load the corresponding data, run the inference pipeline, and display the results in a table.</li>
    </ol>

    <h2>License</h2>
    <p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for details.</p>

</body>
</html>
