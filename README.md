# AI-Powered Parts Optimization Platform (POP)

An intelligent e-commerce platform featuring dynamic inventory management, user and admin management, demand forecasting, and sales insights powered by machine learning.

## Project Overview

This is an AI-powered Parts Optimization Platform that demonstrates real-time dynamic updates, demand forecasting, admin control, and OTP-secured checkout. The platform combines Flask web development with machine learning algorithms to provide intelligent inventory management and sales analytics.

## Features Summary

### Core Features
- **User Registration & Authentication**: Secure user accounts with role-based access
- **Admin Dashboard**: Real-time product management, user administration, and sales monitoring
- **Shopping Cart System**: Dynamic cart management with stock validation
- **OTP-Secured Checkout**: 5-digit OTP verification for secure transactions
- **Dynamic CSV Synchronization**: Real-time data updates across all modules

### ML & Analytics Features
- **EMA-based Demand Forecasting**: Daily-updating exponential moving average predictions
- **SMA/EMA Comparative Visualization**: Side-by-side trend analysis
- **Forecasting Accuracy Metrics**: MAE, RMSE, and MAPE evaluation
- **Bundle Recommendations**: Market basket analysis for cross-selling
- **Inventory Optimization**: Automated reorder suggestions based on demand patterns
- **Sales Insights**: Interactive analytics dashboard with trend analysis

### Admin Features
- **Product Management**: Add, edit, delete products with real-time stock tracking
- **User Management**: Admin PIN-protected user administration
- **Sales Monitoring**: Comprehensive sales reports and transaction history
- **Forecasting Dashboard**: ML-powered demand predictions with visual charts
- **Real-time Updates**: Dynamic CSV synchronization across all modules

## ML Features

### Demand Forecasting
- **EMA (Exponential Moving Average)**: Primary forecasting method that updates daily
- **SMA (Simple Moving Average)**: Comparative baseline for trend analysis
- **Random Forest Models**: Advanced ML models for products with sufficient data
- **Metrics Evaluation**: Comprehensive accuracy assessment with MAE, RMSE, and MAPE

### Analytics Capabilities
- **Trend Analysis**: Identify growing, declining, and stable product trends
- **Bundle Analysis**: Market basket analysis for frequently bought together items
- **Inventory Optimization**: Automated reorder flagging based on demand predictions
- **Sales Analytics**: Interactive charts and insights for business intelligence

## Project Structure

```
pop_project/
├── app.py                          # Main Flask application
├── ml_demand.py                    # ML demand forecasting algorithms
├── bundle_recommender.py           # Market basket analysis
├── insights.py                     # Sales analytics and reporting
├── utils_pop.py                    # Utility functions
├── final_testing_POP.py            # Testing utilities (DO NOT DELETE)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── templates/                      # HTML templates
│   ├── base.html                   # Base template with navigation
│   ├── index.html                  # Homepage
│   ├── login.html                  # User authentication
│   ├── register.html               # User registration
│   ├── cart.html                   # Shopping cart
│   ├── checkout.html               # OTP checkout process
│   ├── sections.html               # Product categories
│   ├── section_products.html       # Products by category
│   ├── bundles.html                # Bundle recommendations
│   ├── insights.html               # Analytics dashboard
│   ├── inventory.html              # Stock management
│   ├── admin.html                  # Admin dashboard
│   ├── admin_users.html            # User management
│   ├── admin_sales.html            # Sales reports
│   ├── admin_pin.html              # PIN verification
│   └── forecast.html                # Forecasting dashboard
│
├── static/                         # Static assets
│   ├── style.css                   # Custom CSS styling
│   └── js/
│       └── insights.js             # Interactive charts
│
├── Data Files (CSV):
├── users.csv                       # User accounts and authentication
├── products.csv                    # Product catalog with pricing and stock
├── sales.csv                       # Transaction history and orders
├── stocks.csv                      # Additional inventory data
├── bundle_suggestions.csv          # Generated bundle recommendations
├── reorder_suggestions.csv         # Generated reorder predictions
├── carts.json                      # Persistent shopping cart storage
├── admin_log.txt                   # Admin action logging
│
└── ML Models (.joblib files):
├── rf_model_*.joblib               # Trained Random Forest models for forecasting
```

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- Git (optional)
- Virtual environment (recommended)

### For Windows Users

1. **Create and activate virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks the activation, run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```powershell
   python app.py
   ```

4. **Access in browser:**
   ```
   http://127.0.0.1:5000
   ```

### For macOS/Linux Users

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access in browser:**
   ```
   http://127.0.0.1:5000
   ```

## Output Files Explanation

### Data Files
- **`sales.csv`**: Purchase logs with transaction history, order IDs, items, quantities, prices, and usernames
- **`products.csv`**: Product details including IDs, names, sections, prices, stock levels, and descriptions
- **`users.csv`**: User and admin credentials with role-based access control
- **`stocks.csv`**: Additional inventory data for comprehensive stock management

### Generated Analytics
- **`reorder_suggestions.csv`**: Restocking recommendations based on demand forecasting and current stock levels
- **`bundle_suggestions.csv`**: Frequently bought together items for cross-selling opportunities
- **`carts.json`**: Persistent shopping cart storage for user sessions
- **`admin_log.txt`**: Admin action logging for security and audit trails

### ML Model Files
- **`rf_model_*.joblib`**: Trained Random Forest models for specific products, enabling accurate demand forecasting

## Testing Instructions

To test all functionalities, run the comprehensive test suite:

```bash
python final_testing_POP.py
```

This will verify:
- User registration and authentication
- Admin dashboard functionality
- Shopping cart operations
- OTP checkout process
- ML forecasting accuracy
- CSV synchronization
- All core features and integrations

## Troubleshooting Section

### Common Issues and Solutions

**Port Conflicts:**
- If port 5000 is in use, the app will show an error
- Solution: Kill the process using the port or change the port in `app.py`

**File Lock Errors:**
- Windows may lock CSV files if they're open in Excel
- Solution: Close all Excel instances and restart the application

**Model Loading Problems:**
- If `.joblib` files are missing, the app will use fallback forecasting
- Solution: Ensure all model files are present in the project directory

**Permission Errors:**
- Ensure the application has write permissions to the project directory
- Avoid running from OneDrive or cloud-synced folders

**Module Import Errors:**
- If you get `ModuleNotFoundError`, ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```

**PowerShell Execution Policy:**
- If virtual environment activation fails, run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

## Demo Instructions

This application demonstrates:

1. **Real-time Dynamic Updates**: All data changes are immediately reflected across the platform
2. **Demand Forecasting**: ML-powered predictions with accuracy metrics and trend analysis
3. **Admin Control**: Comprehensive product and user management with PIN-protected operations
4. **OTP-Secured Checkout**: Secure transaction processing with 5-digit OTP verification
5. **Interactive Analytics**: Dynamic charts and insights for business intelligence

### Default Test Credentials
- **Admin**: username = `admin123`, password = `sriyanshu1`
- **Admin PIN**: `5624` (for user management operations)
- **Demo OTP**: Displayed in the UI for testing (5-digit, 30-second expiry)

## Key Features Demonstrated

- **Dynamic CSV Synchronization**: Real-time updates across all modules
- **ML-Powered Forecasting**: EMA/SMA algorithms with Random Forest models
- **Bundle Recommendations**: Market basket analysis for cross-selling
- **Inventory Optimization**: Automated reorder suggestions
- **Sales Analytics**: Interactive dashboards with trend analysis
- **Secure Transactions**: OTP-based checkout with attempt limiting
- **Admin Management**: PIN-protected user and product administration

---

**Author**: Sriyanshu  
**Project**: AI-Powered Parts Optimization Platform  
**Last Updated**: December 2024





***************************************************************
⚙️ Final Setup Notes (Very Important)

These are critical setup instructions to ensure your Flask project runs smoothly on both Windows and macOS systems.

🪟 For Windows Users
✅ Step-by-step (this will definitely work)

1️⃣ Copy and paste this exact command into PowerShell:

cd "C:\ML project\sriyanshu project\1\POP project\pop_project"


📝 Note: The quotation marks are essential because your folder names contain spaces (ML project and sriyanshu project).

2️⃣ Press Enter, then type:

dir


This will list all files inside the folder, confirming you are in the correct working directory.

🔹 How to Fix PowerShell Script Restriction (If You Get a Security Error)

✅ Option 1 — Temporarily bypass restriction (safe and recommended)

If PowerShell blocks the script during activation, run this in the same terminal window before activating your virtual environment:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


Then, activate your virtual environment again:

.\.venv\Scripts\Activate.ps1

🍎 For macOS Users

1️⃣ Open Terminal, then navigate to your project folder with this command:

cd "/Users/<your-username>/Documents/ML project/sriyanshu project/1/POP project/pop_project"


⚠️ Replace <your-username> with your actual Mac username (you can find it by typing whoami).

2️⃣ Check that you’re in the right folder by typing:

ls


3️⃣ Activate your virtual environment:

source .venv/bin/activate


If you haven’t created one yet, run:

python3 -m venv .venv
source .venv/bin/activate


4️⃣ Finally, to start your Flask app:

python3 app.py


Then open your browser and visit:
👉 http://127.0.0.1:5000