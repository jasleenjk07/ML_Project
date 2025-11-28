# BlinkIt Business Intelligence & Operational Growth (BIOG)

This repository contains a comprehensive business intelligence dashboard for BlinkIt, providing insights and predictions based on operational data.

## Project Overview

The BlinkIt Analysis project is a business intelligence dashboard built with Streamlit that provides insights into various aspects of the business:

1. **Business Overview** - Key metrics and sales trends
2. **Customer Segmentation** - Clustering of customers based on behavior
3. **Product Demand Prediction** - Forecasting future product demand
4. **Delivery Performance Analysis** - Analysis of delivery efficiency
5. **Feedback Analysis** - Analysis of customer sentiment and feedback

The application uses various data science techniques including clustering, regression modeling, and data visualization to extract insights from BlinkIt's operational data.

## Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Basic data visualization
- **Plotly** - Interactive charts and visualizations
- **Scikit-learn** - Machine learning algorithms
  - KMeans for customer segmentation
  - RandomForestRegressor for product demand prediction
- **Streamlit** - Web application framework

## Data Sources

The project uses multiple datasets located in the `blink-it/` directory:

1. **Products Dataset** (`blinkit_products.csv`) - Information about products
2. **Orders Dataset** (`blinkit_orders.csv`) - Customer order information
3. **Order Items Dataset** (`blinkit_order_items.csv`) - Items within each order
4. **Customers Dataset** (`blinkit_customers.csv`) - Customer information
5. **Customer Feedback Dataset** (`blinkit_customer_feedback.csv`) - Ratings and sentiment
6. **Delivery Performance Dataset** (`blinkit_delivery_performance.csv`) - Delivery metrics
7. **Inventory Dataset** (`blinkit_inventoryNew.csv`) - Stock levels
8. **Marketing Performance Dataset** (`blinkit_marketing_performance.csv`) - Campaign performance

## Analysis Components

### Business Overview

- Displays key business metrics (orders, customers, revenue)
- Shows sales trends over time using interactive charts
- Provides a high-level summary of business performance

### Customer Segmentation

- Uses K-means clustering to segment customers based on:
  - Average order value
  - Total spent
  - Order count
- Determines optimal number of clusters using the Elbow Method
- Visualizes customer segments in 3D space
- Provides statistical analysis of each customer segment

### Product Demand Prediction

- Uses Random Forest Regressor to predict future product demand
- Takes time-based features (month, year) into account
- Allows selection of specific products and categories
- Provides model evaluation metrics (RMSE, R²)
- Visualizes historical and predicted demand

### Delivery Performance Analysis

- Analyzes delivery status distribution
- Identifies reasons for delivery delays
- Studies correlation between distance and delivery time
- Determines optimal delivery radius
- Provides hourly breakdown of delivery performance

### Feedback Analysis

- Analyzes distribution of customer ratings
- Performs sentiment analysis on customer feedback
- Identifies problematic categories through average ratings
- Extracts common feedback themes
- Helps prioritize areas for improvement

## How to Run

1. Ensure you have all required libraries installed:
```
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
```

2. Navigate to the project directory:
```
cd blinkit-analysis
```

3. Run the Streamlit app:
```
streamlit run app.py
```

4. The dashboard will open in your default web browser at http://localhost:8501

## Project Structure

```
.
├── blink-it/                       # Data directory
│   ├── blinkit_products.csv
│   ├── blinkit_orders.csv
│   ├── blinkit_order_items.csv
│   ├── blinkit_customers.csv
│   ├── blinkit_customer_feedback.csv
│   ├── blinkit_delivery_performance.csv
│   ├── blinkit_inventoryNew.csv
│   └── blinkit_marketing_performance.csv
│
├── blinkit-analysis/               # Application directory
│   └── app.py                      # Main Streamlit application
│
└── README.md                       # This file
```

## Business Value

- **Targeted Marketing**: Segment customers for personalized campaigns
- **Inventory Optimization**: Forecast demand to optimize stock levels
- **Delivery Efficiency**: Identify bottlenecks in delivery operations
- **Customer Satisfaction**: Address issues identified through feedback
- **Data-Driven Decisions**: Make strategic decisions based on insights

## Future Enhancements

- Advanced demand forecasting with additional features
- Natural language processing for deeper feedback analysis
- Real-time analytics with live data connections
- Product recommendation system
- Geospatial analysis for delivery optimization 