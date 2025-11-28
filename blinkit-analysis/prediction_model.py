import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

def prepare_sales_data(orders_df, order_items_df, products_df):
    """Prepare sales data for modeling"""

    sales_data = pd.merge(
        order_items_df, 
        orders_df[['order_id', 'order_date']], 
        on='order_id', 
        how='left'
    )
    

    sales_data = pd.merge(
        sales_data, 
        products_df[['product_id', 'product_name', 'category', 'price']], 
        on='product_id', 
        how='left'
    )
    

    sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])
    

    sales_data['month'] = sales_data['order_date'].dt.month
    sales_data['year'] = sales_data['order_date'].dt.year
    sales_data['month_year'] = sales_data['order_date'].dt.strftime('%Y-%m')
    
    return sales_data

def create_product_monthly_sales(sales_data):
    """Aggregate data to monthly sales by product"""

    monthly_sales = sales_data.groupby(
        ['product_id', 'product_name', 'category', 'month_year', 'month', 'year']
    )['quantity'].sum().reset_index()

    monthly_sales = monthly_sales.rename(columns={'quantity': 'total_quantity'})
    
    monthly_sales['date'] = pd.to_datetime(monthly_sales['month_year'])
    monthly_sales = monthly_sales.sort_values(['product_id', 'date'])
    
    return monthly_sales

def create_features(product_data):
    """Create features for prediction"""
    df = product_data.copy()
    df = df.sort_values(['product_id', 'date'])
    
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df.groupby('product_id')['total_quantity'].shift(lag)
    
    # Create seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    

    df = df.fillna(0)
    
    return df

def train_arima_model(time_series):
    """Train an ARIMA model on the time series data"""
    p, d, q = 1, 1, 1 # ARIMA parameters
    
    model = ARIMA(time_series, order=(p, d, q))
    model_fit = model.fit()
    
    return model_fit

def train_forecast_models(product_data):
    """Train multiple models for product demand forecasting"""
    features = [
        'month', 'year', 'lag_1', 'lag_2', 'lag_3',
        'month_sin', 'month_cos'
    ]
    
    target = 'total_quantity'
    
    train_size = int(len(product_data) * 0.8)
    train_data = product_data[:train_size]
    test_data = product_data[train_size:]

    test_dates = test_data['date'].reset_index(drop=True)
    test_actual = test_data[target].values

    X = product_data[features]
    y = product_data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = {}
    
    # Train feature-based models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred)
        }
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'scaler': scaler,
            'features': features,
            'test_predictions': y_pred,
            'test_actual': test_actual,
            'test_dates': test_dates,
            'model_type': 'feature-based'
        }
    
    
    try:
        arima_model = train_arima_model(train_data['total_quantity'])
        
        # Make predictions for test period
        arima_preds = arima_model.forecast(steps=len(test_data))
        
        # Ensure predictions are non-negative
        arima_preds = np.maximum(arima_preds, 0)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(test_actual, arima_preds)),
            'MAE': mean_absolute_error(test_actual, arima_preds),
            'R²': r2_score(test_actual, arima_preds)
        }
        
        results['ARIMA'] = {
            'model': arima_model,
            'metrics': metrics,
            'test_predictions': arima_preds,
            'test_actual': test_actual,
            'test_dates': test_dates,
            'model_type': 'time-series'
        }
    except Exception as e:
        st.warning(f"ARIMA model failed: {e}. Using only ML models.")
    
    return results

def make_predictions(product_data, model_results, future_periods=6):
    """Make future predictions using trained models"""
    last_date = product_data['date'].max()
    
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=future_periods,
        freq='MS'  # Month Start
    )
    
    #DataFrame with time features
    future_df = pd.DataFrame({
        'date': future_dates,
        'month_year': future_dates.strftime('%Y-%m'),
        'month': future_dates.month,
        'year': future_dates.year,
        'month_sin': np.sin(2 * np.pi * future_dates.month/12),
        'month_cos': np.cos(2 * np.pi * future_dates.month/12)
    })
    
    # Get the most recent values for lag features
    recent_values = product_data.tail(3)['total_quantity'].values
    
    # Make predictions for each model
    all_predictions = {}
    
    for model_name, results in model_results.items():
        if results.get('model_type') == 'feature-based':
            model = results['model']
            scaler = results['scaler']
            features = results['features']
            
            # Create features for future prediction
            future_preds = []
            
            for i in range(len(future_df)):
                # Current month features
                current = future_df.iloc[i].copy()
                
                # Add lag features
                if i == 0:
                    lag_1 = recent_values[-1] if len(recent_values) > 0 else 0
                    lag_2 = recent_values[-2] if len(recent_values) > 1 else 0
                    lag_3 = recent_values[-3] if len(recent_values) > 2 else 0
                elif i == 1:
                    lag_1 = future_preds[0]
                    lag_2 = recent_values[-1] if len(recent_values) > 0 else 0
                    lag_3 = recent_values[-2] if len(recent_values) > 1 else 0
                elif i == 2:
                    lag_1 = future_preds[1]
                    lag_2 = future_preds[0]
                    lag_3 = recent_values[-1] if len(recent_values) > 0 else 0
                else:
                    lag_1 = future_preds[i-1]
                    lag_2 = future_preds[i-2]
                    lag_3 = future_preds[i-3]
                
                # Create feature dataframe
                current_data = pd.DataFrame({
                    'month': [current['month']],
                    'year': [current['year']],
                    'lag_1': [lag_1],
                    'lag_2': [lag_2],
                    'lag_3': [lag_3],
                    'month_sin': [current['month_sin']],
                    'month_cos': [current['month_cos']]
                })
                
                # Scale features
                current_scaled = scaler.transform(current_data[features])
                
                # Make prediction
                pred = model.predict(current_scaled)[0]
                pred = max(0, pred)  # Ensure non-negative predictions
                future_preds.append(pred)
            
            future_df[f'pred_{model_name}'] = future_preds
            all_predictions[model_name] = future_preds
            
        elif results.get('model_type') == 'time-series':
            # For ARIMA model
            model = results['model']
            
            # Forecast
            arima_preds = model.forecast(steps=future_periods)
            arima_preds = np.maximum(arima_preds, 0)  # Ensure non-negative
            
            future_df[f'pred_{model_name}'] = arima_preds
            all_predictions[model_name] = arima_preds
    
    return future_df, all_predictions

def display_metrics_table(model_metrics):
    """Display a simple metrics table"""
    # Create DataFrame for display
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df = metrics_df.round(3)
    
    # Apply formatting 
    styled_df = metrics_df.style.highlight_min(props='color:white; background-color:green;', axis=0, subset=['RMSE', 'MAE'])
    styled_df = styled_df.highlight_max(props='color:white; background-color:green;', axis=0, subset=['R²'])
    
    return styled_df

def product_demand_prediction_app(data):
    """Main function for the product demand prediction page"""
    st.title("📈 Product Demand Prediction")
    st.write("This tool predicts future product demand to help with inventory planning.")
    
    # Prepare data
    with st.spinner("Processing sales data..."):
        sales_data = prepare_sales_data(data['orders'], data['order_items'], data['products'])
        monthly_sales = create_product_monthly_sales(sales_data)
    
    # Select product
    col1, col2 = st.columns(2)
    with col1:
        categories = sorted(data['products']['category'].unique())
        selected_category = st.selectbox("Select Category", categories)
    
    with col2:
        category_products = data['products'][data['products']['category'] == selected_category]
        product_list = sorted(category_products['product_name'].unique())
        selected_product = st.selectbox("Select Product", product_list)
    
    # Get product data
    selected_product_id = data['products'][data['products']['product_name'] == selected_product]['product_id'].values[0]
    product_data = monthly_sales[monthly_sales['product_id'] == selected_product_id].copy()
    
    if len(product_data) < 5:
        st.warning("⚠️ Not enough data for prediction. Please select a product with more historical data.")
        return
    
    # Create features
    product_data_with_features = create_features(product_data)
    
    # Show historical sales
    fig_hist = px.line(
        product_data, 
        x='date', 
        y='total_quantity',
        title=f'Historical Sales for {selected_product}',
        template='plotly_white'
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    with st.spinner("Training prediction models..."):
        model_results = train_forecast_models(product_data_with_features)
        
        model_metrics = {
            name: {
                'RMSE': results['metrics']['RMSE'],
                'MAE': results['metrics']['MAE'],
                'R²': results['metrics']['R²']
            }
            for name, results in model_results.items()
        }
        
        best_model = min(model_results.items(), key=lambda x: x[1]['metrics']['RMSE'])[0]
        
        st.subheader("Model Performance")
        # st.write("Lower RMSE/MAE and higher R² indicate better models")
        st.dataframe(display_metrics_table(model_metrics))
    
    # Make predictions
    prediction_periods = st.slider("Months to Predict", 1, 12, 6)
    
    with st.spinner("Generating predictions..."):
        future_df, predictions = make_predictions(
            product_data_with_features, 
            model_results,
            future_periods=prediction_periods
        )
        
        # Plot predictions
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=product_data['date'],
            y=product_data['total_quantity'],
            name='Historical Sales',
            line=dict(color='blue', width=2)
        ))
        
        colors = ['red', 'green', 'purple', 'orange']
        for i, model_name in enumerate(model_results.keys()):
            fig.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df[f'pred_{model_name}'],
                name=f'{model_name} Prediction',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=f'Demand Forecast for {selected_product}',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key Insights")
        
        total_predicted = int(sum(predictions[best_model]))
        avg_monthly = int(np.mean(predictions[best_model]))
        peak_month_idx = np.argmax(predictions[best_model])
        peak_month = future_df.iloc[peak_month_idx]['date'].strftime('%B %Y')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predicted Demand", f"{total_predicted} units")
        with col2:
            st.metric("Monthly Average", f"{avg_monthly} units")
        with col3:
            st.metric("Peak Month", peak_month)
        
        # Inventory recommendation
        safety_stock = st.slider("Safety Stock (%)", 10, 50, 20) / 100
        recommended_inventory = total_predicted * (1 + safety_stock)
        
        st.info(
            f"💡 Recommendation: Stock {int(recommended_inventory)} units for the next "
            f"{prediction_periods} months (includes {int(safety_stock*100)}% safety stock)"
        ) 