"""
Used Car Price Prediction - Streamlit Web Application
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-price {
        font-size: 3rem;
        font-weight: bold;
        color: white;
    }
    .feature-importance {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_artifacts():
    """Load trained model and artifacts"""
    try:
        model = joblib.load('best_car_price_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        metrics = joblib.load('model_metrics.pkl')
        return model, label_encoders, scaler, metrics
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the training script first.")
        st.stop()

def detect_model_name(model):
    """Dynamically detect the actual model name"""
    model_type = str(type(model)).lower()
    
    if 'randomforest' in model_type:
        return "Random Forest"
    elif 'xgboost' in model_type or 'xgb' in model_type:
        return "XGBoost"
    elif 'lightgbm' in model_type or 'lgbm' in model_type:
        return "LightGBM"
    elif 'catboost' in model_type:
        return "CatBoost"
    elif 'linear' in model_type or 'regression' in model_type:
        return "Linear Regression"
    elif 'gradientboosting' in model_type:
        return "Gradient Boosting"
    elif 'extratrees' in model_type:
        return "Extra Trees"
    elif 'adaboost' in model_type:
        return "AdaBoost"
    elif 'decisiontree' in model_type:
        return "Decision Tree"
    elif 'svm' in model_type or 'svr' in model_type:
        return "Support Vector Machine"
    elif 'kneighbors' in model_type:
        return "K-Nearest Neighbors"
    else:
        # Extract class name as fallback
        class_name = type(model).__name__
        return class_name.replace('Regressor', '').replace('Classifier', '')

def get_model_details(model):
    """Get detailed model information"""
    model_name = detect_model_name(model)
    details = {"name": model_name}
    
    try:
        if hasattr(model, 'n_estimators'):
            details['estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            details['max_depth'] = model.max_depth
        if hasattr(model, 'learning_rate'):
            details['learning_rate'] = model.learning_rate
        if hasattr(model, 'random_state'):
            details['random_state'] = model.random_state
    except:
        pass
    
    return details

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    except:
        return None

def get_unique_values_from_encoders(label_encoders):
    """Get unique values from label encoders for dropdowns"""
    unique_values = {}
    for col, encoder in label_encoders.items():
        try:
            unique_values[col] = list(encoder.classes_)
        except:
            unique_values[col] = []
    return unique_values

def main():
    # Header
    st.markdown('<div class="main-header">🚗 Used Car Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Price Estimation using Machine Learning</div>', unsafe_allow_html=True)
    
    # Load model and artifacts
    model, label_encoders, scaler, metrics = load_model_and_artifacts()
    
    # Dynamically detect model information
    model_details = get_model_details(model)
    model_name = model_details["name"]
    
    # Get unique values for dropdowns
    unique_values = get_unique_values_from_encoders(label_encoders)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=80)
        st.title("📊 Model Information")
        
        st.markdown("---")
        
        # Model metrics
        st.subheader("🎯 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{metrics['test_r2']:.3f}")
            st.metric("RMSE", f"${metrics['test_rmse']:,.0f}")
        with col2:
            st.metric("MAE", f"${metrics['test_mae']:,.0f}")
            st.metric("Training Time", f"{metrics['training_time']:.1f}s")
        
        st.markdown("---")
        
        # Model info
        st.subheader("🤖 Model Details")
        st.info(f"**Algorithm:** {model_name}")
        
        # Show model parameters if available
        param_text = ""
        if 'estimators' in model_details:
            param_text += f"Estimators: {model_details['estimators']}\n"
        if 'max_depth' in model_details:
            param_text += f"Max Depth: {model_details['max_depth']}\n"
        if 'learning_rate' in model_details:
            param_text += f"Learning Rate: {model_details['learning_rate']}\n"
        
        if param_text:
            st.code(param_text.strip())
        
        st.info("**Features:** 10 engineered features")
        
        st.markdown("---")
        
        st.caption("Built with ❤️ using Python & Streamlit")
    
    # Main content
    st.header("Enter Car Details for Price Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Brand
        brands = unique_values.get('brand', [])
        brand = st.selectbox(
            "🚗 Brand",
            options=brands,
            index=brands.index('Ford') if 'Ford' in brands else 0,
            help="Select the car brand"
        )
        
        # Model Year
        current_year = datetime.now().year
        model_year = st.slider(
            "📅 Model Year",
            min_value=1990,
            max_value=current_year,
            value=2020,
            help="Year of manufacture"
        )
        
        # Fuel Type
        fuel_types = unique_values.get('fuel_type', [])
        fuel_type = st.selectbox(
            "⛽ Fuel Type",
            options=fuel_types,
            index=fuel_types.index('Gasoline') if 'Gasoline' in fuel_types else 0,
            help="Type of fuel"
        )
    
    with col2:
        # Model
        models = unique_values.get('model', [])
        car_model = st.selectbox(
            "🏎️ Model",
            options=models,
            index=0,
            help="Select the car model"
        )
        
        # Milage
        milage = st.number_input(
            "🛣️ Mileage (miles)",
            min_value=0,
            max_value=500000,
            value=30000,
            step=1000,
            help="Total miles driven"
        )
        
        # Transmission
        transmissions = unique_values.get('transmission', [])
        transmission = st.selectbox(
            "⚙️ Transmission",
            options=transmissions,
            index=0,
            help="Transmission type"
        )
    
    with col3:
        # Accident History
        accidents = unique_values.get('accident', [])
        accident = st.selectbox(
            "🚨 Accident History",
            options=accidents,
            index=accidents.index('None reported') if 'None reported' in accidents else 0,
            help="Accident history"
        )
        
        # Clean Title
        clean_titles = unique_values.get('clean_title', [])
        clean_title = st.selectbox(
            "📋 Clean Title",
            options=clean_titles,
            index=clean_titles.index('Yes') if 'Yes' in clean_titles else 0,
            help="Whether car has clean title"
        )
    
    st.markdown("---")
    
    # Calculate derived features
    car_age = current_year - model_year
    milage_per_year = milage / max(car_age, 1)
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Encode inputs
            brand_encoded = label_encoders['brand'].transform([brand])[0]
            model_encoded = label_encoders['model'].transform([car_model])[0]
            fuel_type_encoded = label_encoders['fuel_type'].transform([fuel_type])[0]
            transmission_encoded = label_encoders['transmission'].transform([transmission])[0]
            accident_encoded = label_encoders['accident'].transform([accident])[0]
            clean_title_encoded = label_encoders['clean_title'].transform([clean_title])[0]
            
            # Create input dataframe with all features
            input_data = pd.DataFrame({
                'brand': [brand_encoded],
                'model': [model_encoded],
                'model_year': [model_year],
                'milage_numeric': [milage],
                'fuel_type': [fuel_type_encoded],
                'transmission': [transmission_encoded],
                'accident': [accident_encoded],
                'clean_title': [clean_title_encoded],
                'car_age': [car_age],
                'milage_per_year': [milage_per_year]
            })
            
            # Scale the features
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            with st.spinner(f'🔄 Calculating price using {model_name}...'):
                prediction = model.predict(input_scaled)[0]
            
            # Ensure prediction is positive
            prediction = max(0, prediction)
            
            # Display prediction
            st.markdown("---")
            st.success("✅ Prediction Complete!")
            
            # Prediction box
            col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
            with col_pred2:
                st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: white; margin: 0;">Estimated Market Price</h3>
                        <div class="prediction-price">${prediction:,.0f}</div>
                        <p style="color: white; margin-top: 1rem;">Powered by {model_name}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Confidence interval
            confidence_interval = metrics['test_rmse'] * 0.67
            price_low = max(0, prediction - confidence_interval)
            price_high = prediction + confidence_interval
            
            # Additional info columns
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Model Confidence", f"{metrics['test_r2']:.1%} R²")
            
            with col_info2:
                st.metric("Expected Range", f"${price_low:,.0f} - ${price_high:,.0f}")
            
            with col_info3:
                st.metric("Car Age", f"{car_age} years")
            
            # Feature importance visualization
            feature_names = ['brand', 'model', 'model_year', 'milage', 'fuel_type', 
                           'transmission', 'accident', 'clean_title', 'car_age', 
                           'milage_per_year']
            
            importance_df = get_feature_importance(model, feature_names)
            
            if importance_df is not None:
                st.markdown("---")
                st.subheader("📊 Feature Importance")
                
                fig = px.bar(
                    importance_df.head(8),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'Top Features Influencing Price ({model_name})',
                    labels={'importance': 'Importance', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Feature importance as determined by the {model_name} algorithm")
            
            # Input summary
            with st.expander("📋 View Input Summary & Analysis"):
                col_sum1, col_sum2 = st.columns(2)
                
                with col_sum1:
                    st.write("**Car Specifications:**")
                    summary_data = {
                        'Feature': ['Brand', 'Model', 'Model Year', 'Car Age', 'Mileage', 
                                  'Mileage/Year', 'Fuel Type', 'Transmission', 'Accident History', 'Clean Title'],
                        'Value': [brand, car_model, str(model_year), f"{car_age} years", 
                                f"{milage:,} mi", f"{milage_per_year:,.0f} mi/year",
                                fuel_type, transmission, accident, clean_title]
                    }
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                
                with col_sum2:
                    st.write("**Price Analysis:**")
                    analysis_data = {
                        'Metric': ['Predicted Price', 'Model Confidence', 'Price Range', 'Model Used'],
                        'Value': [f"${prediction:,.2f}", f"{metrics['test_r2']:.2%} R²", 
                                f"${price_low:,.0f} - ${price_high:,.0f}", model_name]
                    }
                    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True, hide_index=True)
                    
                    st.info(f"""
                    **💡 Powered by {model_name}:**
                    - Lower mileage generally increases value
                    - Newer model years command higher prices  
                    - Clean title and no accidents significantly boost value
                    - Regular maintenance records can increase price
                    """)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please ensure all inputs are valid and try again.")
    
    # Model information section
    with st.expander("🔍 About the AI Model"):
        st.subheader(f"🤖 {model_name} Algorithm")
        
        if model_name == "Random Forest":
            st.write("""
            **Random Forest** is an ensemble learning method that operates by constructing multiple decision trees 
            and outputting the mean prediction of the individual trees. It's particularly effective for:
            - Handling complex non-linear relationships
            - Robust to overfitting
            - Providing feature importance scores
            - Working well with both numerical and categorical data
            """)
        elif model_name == "XGBoost":
            st.write("""
            **XGBoost** (Extreme Gradient Boosting) is an advanced implementation of gradient boosting that:
            - Provides state-of-the-art performance on many datasets
            - Includes regularization to prevent overfitting
            - Handles missing values automatically
            - Offers excellent computational efficiency
            """)
        elif model_name == "LightGBM":
            st.write("""
            **LightGBM** is a gradient boosting framework that uses tree-based learning algorithms and:
            - Provides faster training speed and higher efficiency
            - Uses histogram-based algorithms
            - Requires lower memory usage
            - Supports parallel and GPU learning
            """)
        elif model_name == "CatBoost":
            st.write("""
            **CatBoost** is a machine learning algorithm that uses gradient boosting on decision trees and:
            - Excellently handles categorical features
            - Redoves the need for extensive data preprocessing
            - Provides high accuracy with default parameters
            - Resistant to overfitting
            """)
        else:
            st.write(f"""
            **{model_name}** is the selected machine learning algorithm that has been trained on historical car data 
            to accurately predict market prices based on various vehicle features and market trends.
            """)
        
        st.metric("Model Performance (R²)", f"{metrics['test_r2']:.3f}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🚗 Car Price Prediction System | Powered by {model_name}</p>
            <p>Built with Python, Scikit-learn, and Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()