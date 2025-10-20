# 🚗 Used Car Price Prediction

An end-to-end machine learning project that predicts used car prices using Python, Scikit-learn, and Streamlit.

## 🎯 Features

- **Multiple ML Models**: Linear Regression, Decision Tree, Random Forest
- **Interactive Web App**: Built with Streamlit
- **Beautiful Visualizations**: Using Plotly
- **Model Performance Metrics**: R², RMSE, MAE
- **Real-time Predictions**: Enter car details and get instant price estimates

## 📊 Dataset
Download the dataset from Kaggle:
[Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)

Place the CSV file in the project root as `car_data.csv`.

## 🚀 Installation

1. **Clone the repository**
git clone <your-repo-url>
cd car_price_prediction

2. **Create virtual environment** (recommended)
python -m venv venv
venv\Scripts\activate

3. **Install dependencies**
pip install -r requirements.txt

## 📝 Usage

### Step 1: Train the Model
python model_training.py

This will:
- Load and explore the dataset
- Preprocess data and encode categorical variables
- Train multiple models (Linear Regression, Decision Tree, Random Forest)
- Compare model performance
- Save the best model and encoders

**Output files:**
- `car_price_model.pkl` - Trained model
- `label_encoders.pkl` - Categorical encoders
- `model_metrics.pkl` - Performance metrics

### Step 2: Run the Streamlit App
streamlit run app.py

The app will open in your browser at `http://localhost:8501`

## 🎨 App Features

### 🔮 Predict Price Tab
- Enter car details (name, year, kms driven, fuel type, etc.)
- Get instant price predictions
- View confidence intervals and price ranges

### 📊 Statistics Tab
- Dataset overview
- Statistical summaries
- Sample data preview

### 📈 Visualizations Tab
- Price distribution histograms
- Scatter plots (Price vs Year, Price vs Kms)
- Fuel type and transmission distributions
- Interactive Plotly charts

## 📁 Project Structure
car_price_prediction/
├── app.py                    # Streamlit web application
├── model_training.py         # Model training script
├── car_data.csv             # Dataset
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── car_price_model.pkl     # Trained model (generated)
├── label_encoders.pkl      # Encoders (generated)
└── model_metrics.pkl       # Metrics (generated)

## 🛠️ Technologies Used
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Streamlit** - Web application
- **Plotly** - Interactive visualizations
- **Joblib** - Model persistence

## 📈 Model Performance
The project trains three models and selects the best one:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor** (typically best performer)

## Performance metrics:
- R² Score: ~0.85-0.95
- RMSE: ~1.0-2.0 Lakhs
- MAE: ~0.8-1.5 Lakhs

## 🎯 Input Features
- **Car Name**: Model and brand
- **Year**: Year of manufacture
- **Present Price**: Current showroom price (in Lakhs)
- **Kms Driven**: Total kilometers driven
- **Fuel Type**: Petrol, Diesel, or CNG
- **Seller Type**: Dealer or Individual
- **Transmission**: Manual or Automatic
- **Owner**: Number of previous owners

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is open source and available under the MIT License.

## 👨‍💻 Author - Apurv Saktepar

## 🙏 Acknowledgments
- Dataset from Kaggle
- Streamlit community for excellent documentation
- Scikit-learn for powerful ML tools