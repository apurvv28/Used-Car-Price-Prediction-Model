import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
import re
from datetime import datetime

warnings.filterwarnings('ignore')

def clean_price(price_str):
    """Convert price string to float"""
    if pd.isna(price_str):
        return None
    price_str = str(price_str).replace('$', '').replace(',', '')
    try:
        return float(price_str)
    except:
        return None

def clean_milage(milage_str):
    """Convert milage string to integer"""
    if pd.isna(milage_str):
        return None
    milage_str = str(milage_str).replace(',', '').replace('mi.', '').strip()
    try:
        return int(milage_str)
    except:
        return None

def load_and_clean_data(filepath):
    """Load and clean dataset efficiently"""
    print("📊 Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"   Original data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    df['price_numeric'] = df['price'].apply(clean_price)
    df['milage_numeric'] = df['milage'].apply(clean_milage)
    
    df = df.dropna(subset=['price_numeric', 'milage_numeric'])
    
    Q1 = df['price_numeric'].quantile(0.05)
    Q3 = df['price_numeric'].quantile(0.95)
    df = df[(df['price_numeric'] >= Q1) & (df['price_numeric'] <= Q3)]
    
    current_year = datetime.now().year
    df['car_age'] = current_year - df['model_year']
    df['milage_per_year'] = df['milage_numeric'] / np.maximum(df['car_age'], 1)
    
    categorical_cols = ['fuel_type', 'accident', 'clean_title']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    print(f"   Cleaned data: {df.shape[0]} rows")
    return df

def prepare_features_simple(df):
    """Prepare features with simple encoding"""
    features = ['brand', 'model', 'model_year', 'milage_numeric', 
                'fuel_type', 'transmission', 'accident', 'clean_title',
                'car_age', 'milage_per_year']
    
    df_clean = df[features + ['price_numeric']].copy()
    
    categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'accident', 'clean_title']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
    
    X = df_clean[features]
    y = df_clean['price_numeric']
    
    return X, y, label_encoders

def train_fast_models(X_train, X_test, y_train, y_test):
    """Train models quickly with reasonable defaults"""
    print("\n🚀 Training Models...")
    print("=" * 50)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=20, 
            random_state=42,
            n_jobs=-1
        )
    }
    
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        print("   ✓ XGBoost loaded")
    except ImportError:
        print("   ⚠ XGBoost not available")
    
    try:
        from lightgbm import LGBMRegressor
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        print("   ✓ LightGBM loaded")
    except ImportError:
        print("   ⚠ LightGBM not available")
    
    try:
        from catboost import CatBoostRegressor
        models['CatBoost'] = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        print("   ✓ CatBoost loaded")
    except ImportError:
        print("   ⚠ CatBoost not available")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        start_time = datetime.now()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        results[name] = {
            'model': model,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'training_time': training_time
        }
        
        print(f"   ✅ R²: {test_r2:.4f}")
        print(f"   ✅ RMSE: ${test_rmse:,.2f}")
        print(f"   ✅ MAE: ${test_mae:,.2f}")
        print(f"   ⏱ Time: {training_time:.1f}s")
    
    return results

def select_and_save_best_model(results):
    """Select best model and save it"""
    print("\n🏆 Model Comparison")
    print("=" * 50)
    
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'R²': f"{metrics['test_r2']:.4f}",
            'RMSE': f"${metrics['test_rmse']:,.2f}",
            'MAE': f"${metrics['test_mae']:,.2f}",
            'Time': f"{metrics['training_time']:.1f}s"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n{comparison_df.to_string(index=False)}")
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    best_metrics = results[best_model_name]
    
    print(f"\n🎯 BEST MODEL: {best_model_name}")
    print(f"   Final R²: {best_metrics['test_r2']:.4f}")
    print(f"   Final RMSE: ${best_metrics['test_rmse']:,.2f}")
    print(f"   Final MAE: ${best_metrics['test_mae']:,.2f}")
    
    model_filename = 'best_car_price_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"\n💾 Model saved as: {model_filename}")
    
    return best_model, best_model_name, best_metrics

def install_missing_packages():
    """Helper function to install missing packages"""
    packages = ['xgboost', 'lightgbm', 'catboost']
    missing = []
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n📦 Missing packages: {', '.join(missing)}")
        print("   Install with: pip install " + " ".join(missing))
        print("   For better performance, install all advanced models!")

def main():
    """Fast and efficient training pipeline"""
    print("\n" + "="*60)
    print("🚗 CAR PRICE PREDICTION")
    print("="*60)
    
    install_missing_packages()
    
    df = load_and_clean_data('car_data.csv')
    
    X, y, label_encoders = prepare_features_simple(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📈 Data split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing:  {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = train_fast_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    if not results:
        print("❌ No models were trained successfully!")
        return
    
    best_model, best_model_name, best_metrics = select_and_save_best_model(results)
    
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(best_metrics, 'model_metrics.pkl')
    
    print("\n✅ TRAINING COMPLETED!")
    print("="*50)
    print(f"🎯 Best Model: {best_model_name}")
    print(f"📊 R² Score: {best_metrics['test_r2']:.4f}")
    print(f"💰 RMSE: ${best_metrics['test_rmse']:,.2f}")
    print(f"📏 MAE: ${best_metrics['test_mae']:,.2f}")
    
    print(f"\n💾 Saved files:")
    print(f"   • best_car_price_model.pkl")
    print(f"   • label_encoders.pkl") 
    print(f"   • scaler.pkl")
    print(f"   • model_metrics.pkl")

if __name__ == "__main__":
    main()