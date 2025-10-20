# 🚗 Used Car Price Prediction

An end-to-end machine learning project that predicts used car prices using advanced ensemble models and Streamlit.

## 🎯 Features

- **Advanced ML Models**: Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost
- **Interactive Web App**: Beautiful Streamlit interface with real-time predictions
- **Feature Engineering**: Automatic feature creation (car age, mileage per year)
- **Model Comparison**: Automatic selection of best performing model
- **Beautiful Visualizations**: Interactive Plotly charts and feature importance
- **Professional UI**: Modern, responsive design with gradient styling

## 📊 Dataset

The project uses a comprehensive used car dataset with the following features:
- **Brand & Model**: Vehicle manufacturer and specific model
- **Model Year**: Year of manufacture
- **Mileage**: Total miles driven
- **Fuel Type**: Gasoline, Hybrid, Electric, Diesel, E85 Flex Fuel
- **Engine Specifications**: Horsepower, displacement, cylinders
- **Transmission**: Automatic, Manual, CVT
- **Accident History**: Clean record or reported accidents
- **Clean Title**: Whether vehicle has a clean title
- **Price**: Target variable for prediction

## 🚀 Quick Start

### 1. Clone the Repository

git clone https://github.com/apurvv28/Used-Car-Price-Prediction-Model/tree/main

cd car_price_prediction
2. Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Add Your Dataset
Place your used_cars.csv file in the project root directory.
5. Train the Model
python model_training.py
6. Launch the Web App
streamlit run app.py
The app will open at http://localhost:8501

📝 Detailed Usage
Model Training (model_training.py)

## The training script performs:
Data Loading & Cleaning
Price and mileage normalization
Handling missing values
Outlier detection and removal
Feature Engineering
Car age calculation
Mileage per year
Categorical encoding

## Model Training & Comparison
Linear Regression (baseline)
Random Forest (robust ensemble)
XGBoost (high performance)
LightGBM (fast training)
CatBoost (excellent with categorical features)
Automatic Model Selection
Compares R², RMSE, MAE, and training time
Selects best performing model
Saves model and preprocessing artifacts

- Generated Files:
best_car_price_model.pkl - Best performing trained model
label_encoders.pkl - Categorical variable encoders
scaler.pkl - Feature scaler
model_metrics.pkl - Performance metrics

- Web Application (app.py)
Features:
🎯 Real-time Price Predictions - Instant estimates as you input car details
📊 Model Performance Dashboard - R², RMSE, MAE metrics
🔍 Feature Importance - Visualize what factors drive prices
📈 Interactive Charts - Plotly-powered visualizations
📱 Responsive Design - Works on all devices
🎨 Professional UI - Modern gradient styling

- Input Fields:
Vehicle brand and model
Model year and current mileage
Fuel type and transmission
Accident history and title status
Automatic calculation of car age and mileage/year

📁 Project Structure

car_price_prediction/
├── app.py                    # Streamlit web application
├── model_training.py         # Advanced model training script
├── used_cars.csv            # Dataset (add your file)
├── requirements.txt          # Python dependencies
├── README.md                # This documentation
│
├── best_car_price_model.pkl # Best model (generated)
├── label_encoders.pkl       # Categorical encoders (generated)
├── scaler.pkl              # Feature scaler (generated)
└── model_metrics.pkl       # Performance metrics (generated)

## 🛠️ Technology Stack
Python 3.8+ - Core programming language
Pandas & NumPy - Data manipulation and numerical computing
Scikit-learn - Machine learning algorithms and preprocessing
XGBoost, LightGBM, CatBoost - Advanced ensemble methods
Streamlit - Web application framework
Plotly - Interactive data visualizations
Joblib - Model serialization and persistence

## 📈 Model Performance
Supported Algorithms:
Linear Regression - Fast baseline model
Random Forest - Robust ensemble with feature importance
XGBoost - State-of-the-art gradient boosting
LightGBM - High-speed gradient boosting
CatBoost - Superior categorical feature handling

## Typical Performance Metrics:
R² Score: 0.85 - 0.95 (85-95% variance explained)
RMSE: $2,000 - $8,000 (Root Mean Square Error)
MAE: $1,500 - $6,000 (Mean Absolute Error)
Training Time: 1-30 seconds (depending on model complexity)

- 🎯 Key Features

## Data Preprocessing
Automatic handling of price formats ($10,000 → 10000)
Mileage normalization (50,000 mi. → 50000)
Intelligent outlier detection
Missing value imputation

## Feature Engineering
Car Age: Current year - model year
Mileage/Year: Annual usage pattern
Brand Popularity: Market presence indicator
Categorical Encoding: Smart label encoding

## Model Advantages
Automatic Selection: Chooses best algorithm for your data
Robust Performance: Handles various car types and price ranges
Fast Inference: Real-time predictions
Explainable AI: Feature importance visualization

## 🤝 Contributing
We welcome contributions! Please feel free to:
Fork the repository
Create a feature branch
Submit a Pull Request
Add tests for new functionality
Update documentation

## 📄 License
This project is open source and available under the MIT License.

## 👨‍💻 Author
Apurv Saktepar

## 🙏 Acknowledgments
Dataset providers and automotive industry sources
Streamlit team for excellent documentation
Scikit-learn, XGBoost, LightGBM, and CatBoost developers
Open-source community for continuous improvement

## 🔮 Future Enhancements
Image-based car condition assessment
Regional price variations
Seasonal demand patterns
Integration with automotive APIs
Mobile app version
Advanced feature engineering
Model interpretability (SHAP values)
Automated hyperparameter tuning

⭐ If you find this project useful, please give it a star! ⭐