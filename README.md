# BlinkIt Analytics — BIOG (Business Intelligence & Operational Growth)

A Streamlit-based business intelligence dashboard for BlinkIt operational data. The app surfaces key metrics, customer and product insights, delivery and feedback analysis, and multi-model product demand forecasting.

## Features

| Section | Description |
|--------|-------------|
| **Overview** | Total orders, customers, revenue, and monthly sales trends |
| **Customer Insights** | Average order value, order frequency, and segment distribution |
| **Product Analysis** | Category breakdown and top-selling products |
| **Delivery Analysis** | Delivery status mix and time vs. distance |
| **Customer Feedback** | Ratings, sentiment, and feedback categories |
| **Product Predictions** | Demand forecasting with Random Forest, XGBoost, and ARIMA |

## Project structure

```
ML_Project/
├── blink-it/                    # Raw CSV datasets (used by the dashboard)
├── blinkit-analysis/
│   ├── app.py                   # Main Streamlit application
│   ├── prediction_model.py      # Forecasting models and UI
│   ├── data_automation_demo.py  # Simple ETL demo (orders → cleaned_orders.csv)
│   └── requirements.txt
├── reports/                     # Methodology and extended documentation
├── BlinkIt_Analysis_Guide.ipynb # Exploratory analysis notebook
└── README.md
```

## Prerequisites

- Python 3.8+
- pip

## Setup

1. Clone or download this repository.
2. Install dependencies from the app directory:

```bash
cd blinkit-analysis
pip install -r requirements.txt
```

3. Place all BlinkIt CSV files in `blink-it/` (see [Dataset files](#dataset-files) below). The dashboard loads data from `../blink-it/` relative to `blinkit-analysis/app.py`.

> **Note:** `blinkit_orders.csv` must be present in `blink-it/` for the app to start. If you only have a copy under `blinkit-analysis/`, copy it into `blink-it/`.

## Run the dashboard

From the `blinkit-analysis` directory:

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Optional: data automation demo

A minimal cleaning script lives in `blinkit-analysis/` and expects `blinkit_orders.csv` in that same folder:

```bash
cd blinkit-analysis
python data_automation_demo.py
```

This prints order/revenue summaries and writes `cleaned_orders.csv`.

## Dataset files

Expected in `blink-it/`:

| File | Purpose |
|------|---------|
| `blinkit_products.csv` | Product catalog |
| `blinkit_orders.csv` | Order headers (date, totals) |
| `blinkit_order_items.csv` | Line items per order |
| `blinkit_customers.csv` | Customer profiles and segments |
| `blinkit_customer_feedback.csv` | Ratings and sentiment |
| `blinkit_delivery_performance.csv` | Delivery metrics |
| `blinkit_inventoryNew.csv` | Inventory levels |
| `blinkit_marketing_performance.csv` | Campaign performance |

## Demand forecasting

The **Product Predictions** section (`prediction_model.py`) builds monthly product-level sales, engineers lag and seasonal features, and compares:

- **Random Forest** (100 trees, max depth 10) and **XGBoost** (100 estimators) — feature-based regression
- **ARIMA** (1, 1, 1) — time-series baseline on monthly quantity

**Training & evaluation:** 80/20 chronological split (last 20% of months held out). Features include 1–3 month lags, month/year, and sine/cosine seasonality. The dashboard picks the **best model per product** (lowest RMSE) and shows RMSE, MAE, and R² in the **Model Performance** table.

## Results & accuracy

### Business snapshot (dataset)

| Metric | Value |
|--------|------:|
| Total orders | 5,000 |
| Total customers | 2,500 |
| Total revenue | ₹11,009,308 |
| Avg. order value (customers) | ₹1,102 |
| Avg. orders per customer | 10.5 |
| Avg. customer rating | 3.34 / 5 |
| Avg. delivery time | 4.4 min |
| Avg. delivery distance | 2.72 km |

**Delivery status**

| Status | Share |
|--------|------:|
| On Time | 69.4% |
| Slightly Delayed | 20.7% |
| Significantly Delayed | 9.9% |

**Customer segments** (even split across ~2,500 customers): Regular, Premium, New, Inactive (~25% each).

**Feedback sentiment:** Neutral 34.8% · Negative 32.8% · Positive 32.4%.

### Forecasting model accuracy

Evaluated on **244 products** with ≥10 months of sales history (same pipeline as the dashboard). Errors are in **units/month** on the held-out months.

**Mean metrics (all products)**

| Model | RMSE ↓ | MAE ↓ | R² |
|-------|-------:|------:|---:|
| ARIMA | **1.91** | **1.66** | −3.71 |
| Random Forest | 1.92 | 1.65 | −3.56 |
| XGBoost | 2.28 | 1.93 | −5.81 |

**Best model per product** (lowest RMSE among the three): median RMSE **1.48** units/month; mean RMSE **1.56**. ARIMA wins most often (120 products), followed by Random Forest (71) and XGBoost (53).

**Example — top seller (Baby Food, 17 months of history)**

| Model | RMSE | MAE | R² | Selected |
|-------|-----:|----:|---:|:--------:|
| ARIMA | **2.22** | **1.92** | −3.13 | ✓ |
| Random Forest | 3.64 | 3.35 | −10.16 | |
| XGBoost | 6.59 | 6.41 | −35.52 | |

> **How to read R²:** Values near **1** mean the model explains most variance on the test months; **0** is no better than predicting the mean; **negative** values often appear when monthly history is short and the 20% test window has only a few points. **RMSE** and **MAE** are the primary comparison metrics in the app (lower is better).

Representative strong fits (from project methodology, products with stable demand) often reach **R² ≈ 0.76–0.89** and RMSE **≈ 8–12** on higher-volume SKUs. Open **Product Predictions** in the dashboard to see metrics for any product you select.

### Metric definitions

| Metric | Meaning |
|--------|---------|
| **RMSE** | Root mean squared error — penalizes large misses (units/month) |
| **MAE** | Mean absolute error — typical forecast error in units |
| **R²** | Share of variance explained on the test period (higher is better) |

## Business outcomes

- Forecast product demand for inventory and procurement planning
- Spot delivery bottlenecks (status, distance, timing)
- Prioritize service improvements from feedback and sentiment
- Use customer segments for targeted marketing

## Technologies

- **Streamlit** — Dashboard UI
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — Random Forest regressor
- **XGBoost** — Gradient boosting regressor
- **statsmodels** — ARIMA forecasting
- **Plotly** — Interactive charts
- **Matplotlib & Seaborn** — Static plots

## Further reading

- [`reports/README.md`](reports/README.md) — Detailed component breakdown
- [`reports/BlinkIt_Project_Methodology.md`](reports/BlinkIt_Project_Methodology.md) — Methodology and analysis notes
- [`BlinkIt_Analysis_Guide.ipynb`](BlinkIt_Analysis_Guide.ipynb) — Notebook walkthrough
