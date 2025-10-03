# 🌌 Ultimate Exoplanet Predictor Hub

Welcome to the Ultimate Exoplanet Predictor Hub - A sophisticated machine learning application developed for the NASA Space Apps Challenge. This project focuses on analyzing and classifying Kepler Objects of Interest (KOIs) using advanced stacking models and machine learning techniques.

## 🚀 Project Overview

This Streamlit-based web application serves as a powerful tool for predicting the disposition of exoplanet candidates using the Kepler Objects of Interest (KOI) catalog. It combines multiple machine learning models in a stacking architecture to provide accurate predictions and insights.

### Key Features
- 🤖 Advanced Machine Learning Stacking Model
- 📊 Interactive Streamlit Web Interface
- 🔮 Real-time Prediction Capabilities
- 📈 Data Visualization
- 💫 Support for Multiple KOI Analysis
- 🎯 Custom Threshold Optimization

## 💻 Tech Stack

- **Python 3.x**
- **ML Libraries:**
  - scikit-learn==1.3.2
  - LightGBM==4.1.0
  - XGBoost
  - CatBoost
- **Data Processing:**
  - Pandas
  - NumPy
  - imbalanced-learn==0.11.0
- **Visualization:**
  - Matplotlib
  - Seaborn
- **Web Interface:**
  - Streamlit
- **Model Serialization:**
  - Joblib

## 🛠️ Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/labib116/Nasa-Space-Apps-Challenge.git
   cd Nasa-Space-Apps-Challenge
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## � Project Structure
```
.
├── new_app.py              # Main Streamlit application
├── kepler_koi.csv         # Kepler Objects of Interest dataset
├── ultimate_exoplanet_model.joblib  # Trained model
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 🎯 Usage

1. **Start the Application:**
   ```bash
   streamlit run new_app.py
   ```

2. **Input Data:**
   - Enter KOI IDs (e.g., K00752.01)
   - Multiple IDs can be entered for batch prediction

3. **View Results:**
   - Prediction results with confidence scores
   - Visualization of predictions
   - Detailed analysis of each KOI

## 🔬 Model Architecture

The model uses a sophisticated stacking approach combining:
- Random Forest Classifier
- LightGBM
- XGBoost
- CatBoost
- Meta-model (Logistic Regression)

### Features Used
- Planetary characteristics
- Stellar parameters
- Orbital properties
- Custom engineered features

## 📊 Data

The project utilizes the Kepler Objects of Interest (KOI) dataset from NASA's Exoplanet Archive, containing detailed information about potential exoplanet candidates discovered by the Kepler Space Telescope.

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA Space Apps Challenge
- NASA Exoplanet Archive
- Kepler Science Team

## 👥 Team

Created by [AstraNova] for the NASA Space Apps Challenge 2023

