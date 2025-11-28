# BlinkIt Analytics - BIOG (Business Intelligence & Operational Growth) System

This application uses machine learning to analyze BlinkIt datasets and provide actionable business insights. The project implements clustering and regression models to identify patterns, classify data, and generate predictions for business decision-making.

## Features

1. **Business Overview** - Key metrics and sales trends
2. **Customer Segmentation** - K-means clustering to segment customers based on behavior
3. **Product Demand Prediction** - Random Forest regression to predict future product demand
4. **Delivery Performance Analysis** - Insights to improve delivery operations
5. **Feedback Analysis** - Customer sentiment and feedback category analysis

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install required packages:

```bash
pip install -r requirements.txt
```

### Running the Application

1. Ensure the BlinkIt dataset files are in the `blink-it` directory (relative to the app.py file)
2. Start the Streamlit app:

```bash
streamlit run app.py
```

3. The application will open in your default web browser

## Dataset Files

The application uses the following datasets:
- blinkit_products.csv
- blinkit_orders.csv
- blinkit_order_items.csv
- blinkit_customers.csv
- blinkit_customer_feedback.csv
- blinkit_delivery_performance.csv
- blinkit_inventoryNew.csv
- blinkit_marketing_performance.csv

## Business Outcomes

The system helps achieve the following business outcomes:
- Predict product demand in future periods
- Identify areas for improvement in delivery operations
- Analyze customer feedback to enhance service quality
- Segment customers for targeted marketing and personalized experiences

## Technologies Used

- Streamlit - Web application framework
- Pandas & NumPy - Data manipulation and analysis
- Scikit-learn - Machine learning algorithms (K-means clustering, Random Forest regression)
- Plotly - Interactive data visualization
- Matplotlib & Seaborn - Additional visualization libraries 