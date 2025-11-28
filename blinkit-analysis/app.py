import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from prediction_model import product_demand_prediction_app


st.set_page_config(
    page_title="BlinkIt Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)


st.title("📊 BlinkIt Business Intelligence Dashboard")
st.markdown("""
    Welcome to the BlinkIt Analytics Dashboard! This tool helps you understand your business better.
    Use the sidebar to navigate between different sections.
""")


@st.cache_data
def load_data():
    with st.spinner('Loading data... This might take a moment.'):
        try:
            products = pd.read_csv('../blink-it/blinkit_products.csv')
            orders = pd.read_csv('../blink-it/blinkit_orders.csv')
            order_items = pd.read_csv('../blink-it/blinkit_order_items.csv')
            customers = pd.read_csv('../blink-it/blinkit_customers.csv')
            feedback = pd.read_csv('../blink-it/blinkit_customer_feedback.csv')
            delivery = pd.read_csv('../blink-it/blinkit_delivery_performance.csv')
            inventory = pd.read_csv('../blink-it/blinkit_inventoryNew.csv')
            marketing = pd.read_csv('../blink-it/blinkit_marketing_performance.csv')
            
            return {
                'products': products,
                'orders': orders,
                'order_items': order_items,
                'customers': customers,
                'feedback': feedback,
                'delivery': delivery,
                'inventory': inventory,
                'marketing': marketing
            }
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None


data = load_data()
if not data:
    st.stop()
else:
    st.success("✅ Data loaded successfully!")


st.sidebar.title("📑 Navigation")
page = st.sidebar.radio(
    "Select a section",
    ["Overview", "Customer Insights", "Product Analysis", "Delivery Analysis", "Customer Feedback", "Product Predictions"]
)


if page == "Overview":
    st.header("📊 Business Overview")
    

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Orders", f"{len(data['orders']):,}")
    with col2:
        st.metric("Total Customers", f"{len(data['customers']):,}")
    with col3:
        total_revenue = data['orders']['order_total'].sum()
        st.metric("Total Revenue", f"₹{total_revenue:,.2f}")
    

    st.subheader("Monthly Sales")
    orders_df = data['orders'].copy()
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    orders_df['month_year'] = orders_df['order_date'].dt.strftime('%Y-%m')
    monthly_sales = orders_df.groupby('month_year')['order_total'].sum().reset_index()
    
    fig = px.line(monthly_sales, x='month_year', y='order_total', title='Monthly Sales Trend')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Customer Insights":
    st.header("👥 Customer Insights")
    

    st.subheader("Customer Statistics")
    customer_data = data['customers'].copy()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Order Value", f"₹{customer_data['avg_order_value'].mean():.2f}")
    with col2:
        st.metric("Average Orders per Customer", f"{customer_data['total_orders'].mean():.2f}")
    

    st.subheader("Customer Segments")
    segment_counts = customer_data['customer_segment'].value_counts()
    fig = px.pie(values=segment_counts.values, names=segment_counts.index, title='Customer Segments')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Product Analysis":
    st.header("📦 Product Analysis")
    

    st.subheader("Product Categories")
    product_data = data['products'].copy()
    category_counts = product_data['category'].value_counts()
    fig = px.bar(x=category_counts.index, y=category_counts.values, title='Products by Category')
    st.plotly_chart(fig, use_container_width=True)
    

    st.subheader("Top Selling Products")
    order_items_df = data['order_items'].copy()
    product_sales = order_items_df.groupby('product_id')['quantity'].sum().reset_index()
    product_sales = product_sales.merge(product_data[['product_id', 'product_name']], on='product_id')
    top_products = product_sales.sort_values('quantity', ascending=False).head(10)
    
    fig = px.bar(top_products, x='product_name', y='quantity', title='Top 10 Selling Products')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Delivery Analysis":
    st.header("🚚 Delivery Analysis")
    

    st.subheader("Delivery Status")
    delivery_df = data['delivery'].copy()
    status_counts = delivery_df['delivery_status'].value_counts()
    fig = px.pie(values=status_counts.values, names=status_counts.index, title='Delivery Status')
    st.plotly_chart(fig, use_container_width=True)
    

    st.subheader("Delivery Time vs Distance")
    fig = px.scatter(delivery_df, x='distance_km', y='delivery_time_minutes',
                    color='delivery_status', title='Delivery Time vs Distance')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Customer Feedback":
    st.header("💬 Customer Feedback")
    

    st.subheader("Rating Distribution")
    feedback_df = data['feedback'].copy()
    rating_counts = feedback_df['rating'].value_counts().sort_index()
    fig = px.bar(x=rating_counts.index, y=rating_counts.values, title='Rating Distribution')
    st.plotly_chart(fig, use_container_width=True)
    

    st.subheader("Sentiment Analysis")
    sentiment_counts = feedback_df['sentiment'].value_counts()
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution')
    st.plotly_chart(fig, use_container_width=True)
    

    st.subheader("Feedback by Category")
    category_counts = feedback_df['feedback_category'].value_counts()
    fig = px.bar(x=category_counts.index, y=category_counts.values, title='Feedback by Category')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Product Predictions":
    product_demand_prediction_app(data) 