from __future__ import annotations

import csv
import json
import os
import random
import string
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash

import utils_pop
import bundle_recommender
import ml_demand
import insights


app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

BASE_DIR = os.path.dirname(__file__)
USERS_CSV = os.path.join(BASE_DIR, "users.csv")
PRODUCTS_CSV = os.path.join(BASE_DIR, "products.csv")
SALES_CSV = os.path.join(BASE_DIR, "sales.csv")
CARTS_JSON = os.path.join(BASE_DIR, "carts.json")
BUNDLES_CSV = os.path.join(BASE_DIR, "bundle_suggestions.csv")
REORDER_CSV = os.path.join(BASE_DIR, "reorder_suggestions.csv")
ADMIN_LOG = os.path.join(BASE_DIR, "admin_log.txt")


def _atomic_write_text(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    os.replace(tmp, path)


def _atomic_write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    os.replace(tmp, path)


def _hash_password_scrypt(password: str) -> str:
    # Simple scrypt format to match existing file: scrypt:N:r:p$hexsalt$hexhash
    import os as _os
    import hashlib as _hashlib
    N, r, p = 32768, 8, 1
    salt = _os.urandom(16)
    dk = _hashlib.scrypt(password.encode("utf-8"), salt=salt, n=N, r=r, p=p, maxmem=0, dklen=64)
    return f"scrypt:{N}:{r}:{p}$" + salt.hex() + "$" + dk.hex()


def _verify_password_scrypt(stored: str, password: str) -> bool:
    import hashlib as _hashlib
    try:
        meta, salt_hex, hash_hex = stored.split("$")
        method, N, r, p = meta.split(":")
        if method != "scrypt":
            return False
        N, r, p = int(N), int(r), int(p)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = _hashlib.scrypt(password.encode("utf-8"), salt=salt, n=N, r=r, p=p, maxmem=0, dklen=len(expected))
        return dk == expected
    except Exception:
        return False


def _hash_password(password: str) -> str:
    # Prefer Werkzeug; mark with prefix for detection
    return "werkzeug$" + generate_password_hash(password)


def _verify_password(stored_hash: str, password: str) -> bool:
    if stored_hash.startswith("werkzeug$"):
        return check_password_hash(stored_hash[len("werkzeug$"):], password)
    if stored_hash.startswith("scrypt:"):
        return _verify_password_scrypt(stored_hash, password)
    return False


def init_app_files() -> None:
    # users.csv
    if not os.path.exists(USERS_CSV):
        admin_hash = _hash_password("sriyanshu1")
        _atomic_write_csv(
            USERS_CSV,
            rows=[{"username": "admin123", "password_hash": admin_hash, "role": "admin", "email": ""}],
            fieldnames=["username", "password_hash", "role", "email"],
        )
    # products.csv
    if not os.path.exists(PRODUCTS_CSV):
        _atomic_write_csv(
            PRODUCTS_CSV,
            rows=[
                {"product_id": "P001", "name": "Front Brake", "section": "Brake", "price": 1200, "stock": 12, "description": "Brake pads"},
                {"product_id": "P002", "name": "Summer Tires", "section": "Tyres", "price": 48000, "stock": 18, "description": "Quality summer tires"},
            ],
            fieldnames=["product_id", "name", "section", "price", "stock", "description"],
        )
    # sales.csv
    if not os.path.exists(SALES_CSV):
        _atomic_write_csv(
            SALES_CSV,
            rows=[],
            fieldnames=["Date", "OrderID", "Item", "Quantity", "Price", "Username"],
        )
    # carts.json
    if not os.path.exists(CARTS_JSON):
        _atomic_write_text(CARTS_JSON, json.dumps({}, indent=2))
    # analytics CSV placeholders
    if not os.path.exists(BUNDLES_CSV):
        _atomic_write_csv(BUNDLES_CSV, rows=[], fieldnames=["Product_A", "Product_B", "Count"])
    if not os.path.exists(REORDER_CSV):
        _atomic_write_csv(REORDER_CSV, rows=[], fieldnames=["Item", "Predicted_Demand", "Current_Stock", "Reorder_Flag"])
    # admin log
    if not os.path.exists(ADMIN_LOG):
        _atomic_write_text(ADMIN_LOG, "")


# In-memory controls
_failed_attempts: Dict[str, int] = {}
_locked_until: Dict[str, float] = {}
_otp_state: Dict[str, Dict[str, Any]] = {}
_otp_cooldown: Dict[str, float] = {}


def _load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _save_products_df(df: pd.DataFrame) -> None:
    rows = df.to_dict(orient="records")
    _atomic_write_csv(PRODUCTS_CSV, rows=rows, fieldnames=["product_id", "name", "section", "price", "stock", "description"])


def _next_product_id(df: pd.DataFrame) -> str:
    prefix = "P"
    width = 4
    max_num = 0
    if not df.empty and "product_id" in df.columns:
        for pid in df["product_id"].astype(str):
            if pid.startswith(prefix):
                try:
                    num = int(pid[1:])
                    max_num = max(max_num, num)
                except ValueError:
                    continue
    return f"{prefix}{(max_num + 1):0{width}d}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/bundles")
def bundles():
    # Always refresh/generate bundles first
    csv_path = BUNDLES_CSV
    try:
        bundle_recommender.generate_bundles()
    except Exception:
        pass
    try:
        df = pd.read_csv(csv_path)
        # Get top 10 suggestions
        if not df.empty:
            df = df.head(10)
    except Exception:
        df = pd.DataFrame(columns=["Product_A", "Product_B", "Count", "Confidence", "Lift"])
    
    table = df.to_dict(orient="records")
    return render_template("bundles.html", rows=table)


@app.route("/inventory")
def inventory():
    if not _require_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))
    
    # Always refresh/generate inventory suggestions first
    csv_path = REORDER_CSV
    try:
        ml_demand.inventory_advisor()
    except Exception:
        pass
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.DataFrame(columns=["Item", "Predicted_Demand", "Current_Stock", "Reorder_Flag"])

    # Get products with recent sales data
    products_df = _load_csv(PRODUCTS_CSV)
    sales_df = _load_csv(SALES_CSV)
    
    # Calculate recent sales (last 7 days)
    if not sales_df.empty and 'Date' in sales_df.columns:
        sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
        cutoff_date = datetime.now() - timedelta(days=7)
        recent_sales = sales_df[sales_df['Date'] >= cutoff_date]
        
        # Group by item and sum quantities
        if not recent_sales.empty:
            sales_by_item = recent_sales.groupby('Item')['Quantity'].sum().to_dict()
        else:
            sales_by_item = {}
    else:
        sales_by_item = {}
    
    # Create unified inventory data
    unified_inventory = []
    
    # Get all products from products.csv
    products = products_df.to_dict(orient="records")
    
    for product in products:
        product_name = product['name']
        current_stock = int(product['stock'])
        recent_sales_7d = sales_by_item.get(product_name, 0)
        
        # Get predicted demand from reorder suggestions
        predicted_demand = 0.0
        reorder_suggestion = df[df['Item'] == product_name]
        if not reorder_suggestion.empty:
            predicted_demand = float(reorder_suggestion.iloc[0]['Predicted_Demand'])
        
        # Calculate dynamic reorder flag based on stock vs predicted demand
        reorder_flag = _calculate_reorder_flag(current_stock, predicted_demand)
        
        # Determine status based on stock level
        if current_stock <= 0:
            status = "Out of Stock"
        elif current_stock <= 5:
            status = "Low Stock"
        else:
            status = "In Stock"
        
        unified_inventory.append({
            'product': product_name,
            'current_stock': current_stock,
            'recent_sales_7d': recent_sales_7d,
            'status': status,
            'predicted_demand_7d': round(predicted_demand, 1),
            'reorder_flag': reorder_flag
        })
    
    return render_template("inventory.html", inventory_data=unified_inventory)


def _calculate_reorder_flag(current_stock: int, predicted_demand: float) -> dict:
    """Calculate dynamic reorder flag based on stock vs predicted demand"""
    if current_stock <= 0:
        return {
            'text': 'Reorder Needed',
            'class': 'danger',
            'icon': '⚠️'
        }
    elif current_stock < predicted_demand:
        return {
            'text': 'Reorder Needed',
            'class': 'danger',
            'icon': '⚠️'
        }
    elif current_stock <= predicted_demand + 5:  # Within 5 units of predicted demand
        return {
            'text': 'Stable Stock',
            'class': 'warning',
            'icon': '⚡'
        }
    elif current_stock >= predicted_demand + 20:  # 20+ units above predicted demand
        return {
            'text': 'No Reorder Needed',
            'class': 'success',
            'icon': '✅'
        }
    else:
        return {
            'text': 'Monitor',
            'class': 'info',
            'icon': '👁️'
        }


@app.route("/insights")
def insights_view():
    return render_template("insights.html")


@app.route("/api/insights/top_products")
def api_top_products():
    try:
        # Get parameters
        hours = int(request.args.get('hours', 72))  # Default 3 days
        category = request.args.get('category', 'All')
        
        # Load sales data
        sales_df = _load_csv(SALES_CSV)
        if sales_df.empty:
            return jsonify({"products": [], "total": 0})
        
        # Convert date column
        sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_sales = sales_df[sales_df['Date'] >= cutoff_time]
        
        if recent_sales.empty:
            return jsonify({"products": [], "total": 0})
        
        # Filter by category if specified
        if category != 'All':
            products_df = _load_csv(PRODUCTS_CSV)
            if not products_df.empty:
                category_products = products_df[products_df['section'] == category]['name'].tolist()
                recent_sales = recent_sales[recent_sales['Item'].isin(category_products)]
        
        # Aggregate by product
        product_counts = recent_sales.groupby('Item')['Quantity'].sum().sort_values(ascending=False)
        
        # Get top 5
        top_products = product_counts.head(5)
        total_sales = product_counts.sum()
        
        # Format response
        products = []
        for product, count in top_products.items():
            percentage = (count / total_sales * 100) if total_sales > 0 else 0
            products.append({
                "name": product,
                "count": int(count),
                "percentage": round(percentage, 1)
            })
        
        return jsonify({
            "products": products,
            "total": int(total_sales)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------- Authentication -------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        # Validation
        if len(username) < 1 or len(username) > 8:
            flash("Username must be 1-8 characters.", "danger")
            return redirect(url_for("login"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("login"))
        if len(password) > 10:
            flash("Password cannot exceed 10 characters.", "danger")
            return redirect(url_for("login"))
            
        now = time.time()
        if username in _locked_until and now < _locked_until[username]:
            flash("Account locked. Try again later.", "danger")
            return redirect(url_for("login"))

        # 1) Hardcoded admin check
        if username == "admin123" and password == "sriyanshu1":
            session["username"] = "admin123"
            session["role"] = "admin"
            _failed_attempts[username] = 0
            flash("Logged in as admin.", "success")
            return redirect(url_for("admin_dashboard"))

        # 2) Normal user check against users.csv
        users = _load_csv(USERS_CSV)
        row = users[users["username"] == username].head(1)
        if not row.empty and _verify_password(str(row.iloc[0]["password_hash"]), password):
            session["username"] = username
            session["role"] = "user"
            _failed_attempts[username] = 0
            flash("Logged in successfully.", "success")
            return redirect(url_for("sections"))

        # Failure handling (including wrong admin creds)
        _failed_attempts[username] = _failed_attempts.get(username, 0) + 1
        if username == "admin123" and _failed_attempts[username] >= 3:
            _locked_until[username] = now + 60
        flash("Invalid credentials.", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        email = request.form.get("email", "").strip()
        # validations
        if len(username) < 1 or len(username) > 8:
            flash("Username must be 1-8 characters.", "danger")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("register"))
        if len(password) > 10:
            flash("Password cannot exceed 10 characters.", "danger")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))
        users = _load_csv(USERS_CSV)
        if not users.empty and (users["username"] == username).any():
            flash("Username already exists. Please choose another.", "danger")
            return redirect(url_for("register"))
        # create user
        pwd_hash = _hash_password(password)
        new_row = {"username": username, "password_hash": pwd_hash, "role": "user", "email": email}
        users = pd.concat([users, pd.DataFrame([new_row])], ignore_index=True)
        _atomic_write_csv(USERS_CSV, users.to_dict(orient="records"), ["username", "password_hash", "role", "email"])
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


# ------------- Shopping -------------
SECTIONS = ["Brake", "Tyres", "Engine Oils", "Engine Parts", "Body Kit"]


def _load_carts() -> Dict[str, Any]:
    try:
        with open(CARTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_carts(carts: Dict[str, Any]) -> None:
    _atomic_write_text(CARTS_JSON, json.dumps(carts, indent=2))


def _get_bundle_suggestions(product_name: str, limit: int = 2) -> List[Dict[str, Any]]:
    """Get bundle suggestions for a specific product."""
    try:
        # Generate fresh bundle recommendations
        bundle_recommender.generate_bundles()
        
        # Load the bundle suggestions
        df = pd.read_csv(BUNDLES_CSV)
        if df.empty:
            return []
        
        # Filter for suggestions where Product_A matches the product name
        suggestions = df[df['Product_A'] == product_name].head(limit)
        
        # Convert to list of dictionaries
        return suggestions.to_dict(orient="records")
    except Exception:
        return []


@app.route("/sections")
def sections():
    return render_template("sections.html", sections=SECTIONS)


@app.route("/products/<section>")
def products(section: str):
    df = _load_csv(PRODUCTS_CSV)
    items = df[df["section"] == section].to_dict(orient="records") if not df.empty else []
    return render_template("section_products.html", section=section, items=items)


@app.route("/cart")
def cart_view():
    username = session.get("username")
    if not username:
        flash("Please login.", "warning")
        return redirect(url_for("login"))
    
    # Redirect admin users to admin dashboard
    if session.get("role") == "admin":
        flash("Admin users cannot access cart functionality.", "info")
        return redirect(url_for("admin_dashboard"))
    
    carts = _load_carts()
    cart = carts.get(username, [])
    
    # Get bundle suggestions from session
    bundle_suggestions = session.get("bundle_suggestions", [])
    # Clear bundle suggestions after displaying them
    session.pop("bundle_suggestions", None)
    
    return render_template("cart.html", items=cart, bundle_suggestions=bundle_suggestions)


@app.route("/cart/add", methods=["POST"])
def cart_add():
    username = session.get("username")
    if not username:
        flash("Please login.", "warning")
        return redirect(url_for("login"))
    
    # Redirect admin users to admin dashboard
    if session.get("role") == "admin":
        flash("Admin users cannot access cart functionality.", "info")
        return redirect(url_for("admin_dashboard"))
    product_id = request.form.get("product_id")
    qty = int(request.form.get("quantity", 1))
    
    # Load fresh product data to check current stock
    df = _load_csv(PRODUCTS_CSV)
    prod = df[df["product_id"] == product_id].head(1)
    if prod.empty:
        flash("Product not found.", "danger")
        return redirect(url_for("cart_view"))
    
    item = prod.iloc[0].to_dict()
    current_stock = int(item["stock"])
    
    # Check if product is out of stock
    if current_stock <= 0:
        flash("Product is out of stock.", "danger")
        return redirect(url_for("cart_view"))
    
    # Check if requested quantity exceeds available stock
    if qty > current_stock:
        flash(f"Insufficient stock — only {current_stock} available.", "danger")
        return redirect(url_for("cart_view"))
    
    entry = {
        "product_id": item["product_id"],
        "name": item["name"],
        "price": float(item["price"]),
        "quantity": qty,
    }
    carts = _load_carts()
    cart = carts.get(username, [])
    
    # Check total quantity in cart + new quantity
    existing_qty = 0
    for it in cart:
        if it["product_id"] == entry["product_id"]:
            existing_qty = it["quantity"]
            break
    
    total_qty = existing_qty + qty
    if total_qty > current_stock:
        flash(f"Insufficient stock — only {current_stock} available (you have {existing_qty} in cart).", "danger")
        return redirect(url_for("cart_view"))
    
    # If product already in cart, update quantity
    found = False
    for it in cart:
        if it["product_id"] == entry["product_id"]:
            it["quantity"] += qty
            found = True
            break
    if not found:
        cart.append(entry)
    carts[username] = cart
    _save_carts(carts)
    
    # Get bundle suggestions for the added product
    bundle_suggestions = _get_bundle_suggestions(item["name"])
    session["bundle_suggestions"] = bundle_suggestions
    
    flash("Added to cart.", "success")
    return redirect(url_for("cart_view"))


@app.route("/cart/remove", methods=["POST"])
def cart_remove():
    username = session.get("username")
    if not username:
        flash("Please login.", "warning")
        return redirect(url_for("login"))
    
    # Redirect admin users to admin dashboard
    if session.get("role") == "admin":
        flash("Admin users cannot access cart functionality.", "info")
        return redirect(url_for("admin_dashboard"))
    index = int(request.form.get("index", -1))
    carts = _load_carts()
    cart = carts.get(username, [])
    if 0 <= index < len(cart):
        cart.pop(index)
    carts[username] = cart
    _save_carts(carts)
    return redirect(url_for("cart_view"))


@app.route("/cart/update", methods=["POST"])
def cart_update():
    username = session.get("username")
    if not username:
        flash("Please login.", "warning")
        return redirect(url_for("login"))
    
    # Redirect admin users to admin dashboard
    if session.get("role") == "admin":
        flash("Admin users cannot access cart functionality.", "info")
        return redirect(url_for("admin_dashboard"))
    index = int(request.form.get("index", -1))
    quantity = max(1, int(request.form.get("quantity", 1)))
    carts = _load_carts()
    cart = carts.get(username, [])
    if 0 <= index < len(cart):
        cart[index]["quantity"] = quantity
    carts[username] = cart
    _save_carts(carts)
    return redirect(url_for("cart_view"))


@app.route("/checkout", methods=["GET", "POST"])
def checkout():
    username = session.get("username")
    if not username:
        flash("Please login.", "warning")
        return redirect(url_for("login"))
    
    # Redirect admin users to admin dashboard
    if session.get("role") == "admin":
        flash("Admin users cannot access cart functionality.", "info")
        return redirect(url_for("admin_dashboard"))
    carts = _load_carts()
    cart = carts.get(username, [])
    if not cart:
        flash("Cart is empty.", "warning")
        return redirect(url_for("cart_view"))

    state = _otp_state.get(username)
    now = time.time()
    
    # Check OTP cooldown
    if username in _otp_cooldown and now < _otp_cooldown[username]:
        flash("Too many wrong attempts. Please wait 1 minute before retrying.", "danger")
        return redirect(url_for("checkout"))

    # Auto-generate OTP if not already generated
    if not state:
        otp = str(random.randint(10000, 99999))  # 5-digit OTP
        _otp_state[username] = {"otp": otp, "expires": now + 300, "attempts": 0, "resends": 0}  # 5 minutes expiry
        flash(f"OTP generated (demo): {otp}", "info")
        state = _otp_state.get(username)

    bill: Dict[str, Any] | None = None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "send":
            if state and state.get("resends", 0) >= 2:
                flash("Max resends reached.", "danger")
            else:
                otp = str(random.randint(10000, 99999))  # 5-digit OTP
                _otp_state[username] = {"otp": otp, "expires": now + 30, "attempts": 0, "resends": (state.get("resends", 0) + 1) if state else 0}
                flash(f"OTP sent (demo): {otp}", "info")
            return redirect(url_for("checkout"))
        elif action == "verify":
            code = request.form.get("otp", "")
            
            # Validate OTP format
            if len(code) != 5 or not code.isdigit():
                flash("OTP must be exactly 5 digits.", "danger")
                return redirect(url_for("checkout"))
                
            st = _otp_state.get(username)
            if not st or now > st.get("expires", 0):
                flash("OTP expired, please resend.", "danger")
                return redirect(url_for("checkout"))
            if st.get("attempts", 0) >= 3:
                _otp_cooldown[username] = now + 60  # 1 minute cooldown
                flash("Too many wrong attempts. Please wait 1 minute before retrying.", "danger")
                return redirect(url_for("checkout"))
            st["attempts"] = st.get("attempts", 0) + 1
            if code == st.get("otp"):
                # Final stock check before processing order
                df = _load_csv(PRODUCTS_CSV)
                insufficient_stock = []
                
                for entry in cart:
                    mask = df["product_id"] == entry["product_id"]
                    if mask.any():
                        current_stock = int(df.loc[mask, "stock"].iloc[0])
                        if current_stock < entry["quantity"]:
                            insufficient_stock.append(f"{entry['name']}: only {current_stock} available")
                
                if insufficient_stock:
                    flash(f"Stock changed during checkout: {', '.join(insufficient_stock)}", "danger")
                    return redirect(url_for("checkout"))
                
                # process order
                order_id = uuid.uuid4().hex[:10].upper()
                # deduct stock atomically
                for entry in cart:
                    mask = df["product_id"] == entry["product_id"]
                    if mask.any():
                        df.loc[mask, "stock"] = (pd.to_numeric(df.loc[mask, "stock"], errors="coerce").fillna(0) - entry["quantity"]).clip(lower=0)
                _save_products_df(df)
                
                # Trigger inventory refresh when stock is reduced by purchase
                try:
                    ml_demand.inventory_advisor()
                except Exception:
                    pass
                # write sales rows
                try:
                    sales_df = _load_csv(SALES_CSV)
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_rows = []
                    for entry in cart:
                        new_rows.append({
                            "Date": now_str,
                            "OrderID": order_id,
                            "Item": entry["name"],
                            "Quantity": entry["quantity"],
                            "Price": entry["price"],
                            "Username": username,
                        })
                    sales_df = pd.concat([sales_df, pd.DataFrame(new_rows)], ignore_index=True)
                    _atomic_write_csv(SALES_CSV, sales_df.to_dict(orient="records"), ["Date", "OrderID", "Item", "Quantity", "Price", "Username"])
                except Exception:
                    pass
                # prepare bill and clear cart
                subtotal = sum(e["price"] * e["quantity"] for e in cart)
                bill = {
                    "order_id": order_id,
                    "datetime": now_str,
                    "items": cart,
                    "grand_total": subtotal,
                }
                carts[username] = []
                _save_carts(carts)
                _otp_state.pop(username, None)
                flash(f"Order {order_id} placed!", "success")
                return render_template("checkout.html", cart=[], bill=bill)
            else:
                flash("Invalid OTP.", "danger")
                return redirect(url_for("checkout"))
    return render_template("checkout.html", cart=cart)


# ------------- Admin PIN Gate -------------
@app.route("/admin/pin", methods=["GET", "POST"])
def admin_pin():
    if not _require_admin():
        flash("Admin only.", "danger")
        return redirect(url_for("login"))
    if request.method == "POST":
        pin = request.form.get("pin", "").strip()
        if pin == "5624":
            session["admin_pin_ok"] = True
            flash("PIN verified.", "success")
            return redirect(url_for("admin_users"))
        else:
            flash("Invalid PIN.", "danger")
            return redirect(url_for("admin_pin"))
    return render_template("admin_pin.html")


# ------------- Admin User Management -------------
@app.route("/admin/users")
def admin_users():
    if not _require_admin() or not session.get("admin_pin_ok"):
        flash("Admin PIN required.", "danger")
        return redirect(url_for("admin_pin"))
    users = _load_csv(USERS_CSV)
    return render_template("admin_users.html", users=users.to_dict(orient="records") if not users.empty else [])


@app.route("/admin/users/reset", methods=["POST"])
def admin_reset_password():
    if not _require_admin() or not session.get("admin_pin_ok"):
        return redirect(url_for("admin_pin"))
    username = request.form.get("username")
    # Generate random 10-char password
    temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    temp_hash = _hash_password(temp_password)
    
    # Update users.csv
    users = _load_csv(USERS_CSV)
    mask = users["username"] == username
    if mask.any():
        users.loc[mask, "password_hash"] = temp_hash
        _atomic_write_csv(USERS_CSV, users.to_dict(orient="records"), ["username", "password_hash", "role", "email"])
        _log_admin(f"RESET_PASSWORD {username}")
        flash(f"Temporary password: {temp_password}", "success")
    else:
        flash("User not found.", "danger")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/set", methods=["POST"])
def admin_set_password():
    if not _require_admin() or not session.get("admin_pin_ok"):
        return redirect(url_for("admin_pin"))
    username = request.form.get("username")
    new_password = request.form.get("new_password", "").strip()
    
    if len(new_password) < 6 or len(new_password) > 10:
        flash("Password must be 6-10 characters.", "danger")
        return redirect(url_for("admin_users"))
    
    new_hash = _hash_password(new_password)
    
    # Update users.csv
    users = _load_csv(USERS_CSV)
    mask = users["username"] == username
    if mask.any():
        users.loc[mask, "password_hash"] = new_hash
        _atomic_write_csv(USERS_CSV, users.to_dict(orient="records"), ["username", "password_hash", "role", "email"])
        _log_admin(f"SET_PASSWORD {username}")
        flash("Password updated.", "success")
    else:
        flash("User not found.", "danger")
    return redirect(url_for("admin_users"))


# ------------- Admin Sales View -------------
@app.route("/admin/sales")
def admin_sales():
    if not _require_admin():
        flash("Admin only.", "danger")
        return redirect(url_for("login"))
    sales = _load_csv(SALES_CSV)
    if not sales.empty:
        sales = sales.sort_values("Date", ascending=False)
    return render_template("admin_sales.html", sales=sales.to_dict(orient="records") if not sales.empty else [])


# ------------- Admin -------------
def _require_admin():
    return session.get("role") == "admin"


def _log_admin(action: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(ADMIN_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {session.get('username','?')}: {action}\n")
    except Exception:
        pass


@app.route("/admin")
def admin_dashboard():
    if not _require_admin():
        flash("Admin only.", "danger")
        return redirect(url_for("login"))
    df = _load_csv(PRODUCTS_CSV)
    return render_template("admin.html", products=df.to_dict(orient="records") if not df.empty else [], sections=SECTIONS)


@app.route("/admin/add", methods=["POST"])
def admin_add():
    if not _require_admin():
        return redirect(url_for("login"))
    df = _load_csv(PRODUCTS_CSV)
    new = {
        "product_id": request.form.get("product_id") or _next_product_id(df),
        "name": request.form.get("name", "").strip(),
        "section": request.form.get("section", "").strip(),
        "price": float(request.form.get("price", 0)),
        "stock": int(request.form.get("stock", 0)),
        "description": request.form.get("description", "").strip(),
    }
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    _save_products_df(df)
    _log_admin(f"ADD {new['product_id']}")
    
    # Trigger inventory refresh when new product is added
    try:
        ml_demand.inventory_advisor()
    except Exception:
        pass
    
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/edit", methods=["POST"])
def admin_edit():
    if not _require_admin():
        return redirect(url_for("login"))
    pid = request.form.get("product_id")
    df = _load_csv(PRODUCTS_CSV)
    mask = df["product_id"] == pid
    for col in ["name", "section", "price", "stock", "description"]:
        if col in request.form:
            val = request.form.get(col)
            if col in ("price",):
                df.loc[mask, col] = float(val)
            elif col in ("stock",):
                df.loc[mask, col] = int(val)
            else:
                df.loc[mask, col] = val
    _save_products_df(df)
    _log_admin(f"UPDATE {pid}")
    
    # Trigger inventory refresh when stock is updated
    try:
        ml_demand.inventory_advisor()
    except Exception:
        pass
    
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete", methods=["POST"])
def admin_delete():
    if not _require_admin():
        return redirect(url_for("login"))
    pid = request.form.get("product_id")
    df = _load_csv(PRODUCTS_CSV)
    df = df[df["product_id"] != pid]
    _save_products_df(df)
    _log_admin(f"DELETE {pid}")
    return redirect(url_for("admin_dashboard"))


@app.route("/forecast")
def forecast():
    if not _require_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))
    
    # Get timeframe parameters
    timeframe = request.args.get("timeframe", "3")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    
    # Load fresh data
    try:
        sales_df = _load_csv(SALES_CSV)
        products_df = _load_csv(PRODUCTS_CSV)
        
        # Convert Date column to datetime
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        # Calculate forecast data
        forecast_data = _calculate_forecast(sales_df, products_df, timeframe, start_date, end_date)
        
        return render_template("forecast.html", 
                             forecast_data=forecast_data,
                             timeframe=timeframe,
                             start_date=start_date,
                             end_date=end_date)
    except Exception as e:
        flash(f"Error generating forecast: {str(e)}", "danger")
        return redirect(url_for("admin_dashboard"))


@app.route("/forecast/api")
def forecast_api():
    """API endpoint for dynamic forecast updates"""
    if not _require_admin():
        return jsonify({"error": "Admin access required"}), 403
    
    timeframe = request.args.get("timeframe", "3")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    
    try:
        sales_df = _load_csv(SALES_CSV)
        products_df = _load_csv(PRODUCTS_CSV)
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        forecast_data = _calculate_forecast(sales_df, products_df, timeframe, start_date, end_date)
        return jsonify(forecast_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast/metrics")
def forecast_metrics_download():
    """Download forecast metrics CSV"""
    if not _require_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))
    
    try:
        if os.path.exists("forecast_metrics.csv"):
            return send_file("forecast_metrics.csv", as_attachment=True, download_name="forecast_metrics.csv")
        else:
            flash("No metrics file available. Run forecasting first.", "warning")
            return redirect(url_for("forecast"))
    except Exception as e:
        flash(f"Error downloading metrics: {str(e)}", "danger")
        return redirect(url_for("forecast"))


@app.route("/debug/forecast_status")
def debug_forecast_status():
    """Debug endpoint to return JSON per product with SMA/EMA status"""
    if not _require_admin():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        sales_df = _load_csv(SALES_CSV)
        products_df = _load_csv(PRODUCTS_CSV)
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        unique_products = sales_df['Item'].unique()
        debug_results = []
        
        for product_name in unique_products:
            try:
                forecast_data = ml_demand.generate_forecast_for_item(str(product_name), sales_df)
                
                # Get product sales data
                product_sales = sales_df[sales_df['Item'] == product_name]
                n_days = len(product_sales)
                
                sma_data = forecast_data.get('sma', [])
                ema_data = forecast_data.get('ema', [])
                
                debug_results.append({
                    'product': str(product_name),
                    'n_days': n_days,
                    'sma_last': float(sma_data[-1]) if sma_data else 0.0,
                    'ema_last': float(ema_data[-1]) if ema_data else 0.0,
                    'sma_len': len(sma_data),
                    'ema_len': len(ema_data)
                })
                
            except Exception as e:
                debug_results.append({
                    'product': str(product_name),
                    'n_days': 0,
                    'sma_last': 0.0,
                    'ema_last': 0.0,
                    'sma_len': 0,
                    'ema_len': 0,
                    'error': str(e)
                })
        
        return jsonify(debug_results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/forecast_trend")
def debug_forecast_trend():
    """Debug endpoint to verify forecast trends are not flat"""
    if not _require_admin():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        sales_df = _load_csv(SALES_CSV)
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        unique_products = sales_df['Item'].unique()
        trend_results = []
        
        for product_name in unique_products:
            try:
                forecast_data = ml_demand.generate_forecast_for_item(str(product_name), sales_df)
                
                ema_data = forecast_data.get('ema', [])
                forecast_data_list = forecast_data.get('forecast', [])
                
                # Get last EMA and forecast values
                last_ema = float(ema_data[-1]) if ema_data else 0.0
                forecast_values = [float(x) for x in forecast_data_list if x is not None]
                
                # Check if forecast values are all equal (flat line)
                is_flat = len(set(forecast_values)) <= 1 if forecast_values else True
                
                # Calculate trend direction
                if len(forecast_values) >= 2:
                    trend_direction = "increasing" if forecast_values[-1] > forecast_values[0] else "decreasing" if forecast_values[-1] < forecast_values[0] else "flat"
                else:
                    trend_direction = "insufficient_data"
                
                trend_results.append({
                    'product': str(product_name),
                    'last_ema': last_ema,
                    'forecast_values': forecast_values,
                    'is_flat': is_flat,
                    'trend_direction': trend_direction,
                    'forecast_range': max(forecast_values) - min(forecast_values) if forecast_values else 0.0
                })
                
            except Exception as e:
                trend_results.append({
                    'product': str(product_name),
                    'last_ema': 0.0,
                    'forecast_values': [],
                    'is_flat': True,
                    'trend_direction': 'error',
                    'forecast_range': 0.0,
                    'error': str(e)
                })
        
        return jsonify(trend_results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _calculate_forecast(sales_df: pd.DataFrame, products_df: pd.DataFrame, timeframe: str, start_date: str = "", end_date: str = ""):
    """Enhanced forecast calculation using ML models"""
    from datetime import datetime, timedelta
    import ml_demand
    
    now = datetime.now()
    
    # Determine date range for analysis
    if timeframe == "custom" and start_date and end_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_count = (end_dt - start_dt).days + 1
    else:
        days_count = int(timeframe)
        start_dt = now - timedelta(days=days_count)
        end_dt = now
    
    # Filter sales data for the analysis timeframe
    mask = (sales_df['Date'] >= start_dt) & (sales_df['Date'] <= end_dt)
    filtered_sales = sales_df[mask].copy()
    
    # Get unique products from filtered sales
    if filtered_sales.empty:
        return {
            'products': [],
            'chart_data': {"labels": [], "actual": [], "predicted": []},
            'timeframe_info': {
                'start_date': str(start_dt.strftime("%Y-%m-%d")),
                'end_date': str(end_dt.strftime("%Y-%m-%d")),
                'days_count': int(days_count)
            }
        }
    
    unique_products = filtered_sales['Item'].unique()
    
    # Generate forecasts for each product using enhanced ML
    forecast_results = []
    all_forecasts = {}
    
    for product_name in unique_products:
        try:
            # Use the enhanced ML forecasting
            forecast_data = ml_demand.generate_forecast_for_item(str(product_name), sales_df)
            
            # Calculate actual sales in the analysis period
            product_sales = filtered_sales[filtered_sales['Item'] == product_name]
            actual_quantity = int(product_sales['Quantity'].sum())
            avg_price = float(product_sales['Price'].mean()) if not product_sales.empty else 0.0
            
            # Get forecasted quantities (7-day predictions)
            forecasted_quantity = forecast_data.get('total_predicted_demand', 0.0)
            
            # Calculate trend badge
            trend = forecast_data.get('trend', 'Stable')
            if trend == "Increasing":
                trend_badge = "Growing"
            elif trend == "Decreasing":
                trend_badge = "Declining"
            else:
                trend_badge = "Stable"
            
            forecast_results.append({
                'product': str(product_name),
                'actual_sales': actual_quantity,
                'forecasted_sales': float(round(forecasted_quantity, 1)),
                'actual_revenue': float(round(actual_quantity * avg_price, 2)),
                'forecasted_revenue': float(round(forecasted_quantity * avg_price, 2)),
                'daily_avg': float(round(forecasted_quantity / 7, 2)),
                'trend': trend_badge,
                'model_type': forecast_data.get('model_type', 'unknown'),
                'metrics': forecast_data.get('metrics', {}),
                'clipped_flag': forecast_data.get('clipped_flag', False),
                'insufficient_history': forecast_data.get('insufficient_history', False)
            })
            
            # Store predictions and SMA/EMA data for chart
            all_forecasts[product_name] = {
                'forecasted_quantity': forecasted_quantity,
                'sma': forecast_data.get('sma', []),
                'ema': forecast_data.get('ema', [])
            }
            
        except Exception as e:
            print(f"Error forecasting {product_name}: {e}")
            # Fallback to simple calculation
            product_sales = filtered_sales[filtered_sales['Item'] == product_name]
            actual_quantity = int(product_sales['Quantity'].sum())
            avg_price = float(product_sales['Price'].mean()) if not product_sales.empty else 0.0
            daily_avg = actual_quantity / days_count if days_count > 0 else 0.0
            forecasted_quantity = daily_avg * 7  # 7-day forecast
            
            forecast_results.append({
                'product': str(product_name),
                'actual_sales': actual_quantity,
                'forecasted_sales': float(round(forecasted_quantity, 1)),
                'actual_revenue': float(round(actual_quantity * avg_price, 2)),
                'forecasted_revenue': float(round(forecasted_quantity * avg_price, 2)),
                'daily_avg': float(round(daily_avg, 2)),
                'trend': "Stable",
                'model_type': "fallback",
                'metrics': {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
            })
    
    # Prepare chart data with SMA and EMA lines
    chart_data = {"labels": [], "actual": [], "forecast": [], "sma": [], "ema": []}
    
    # Create date range: last 7 days (actual) + next 7 days (forecast)
    chart_start = now - timedelta(days=7)
    
    for i in range(14):
        date = chart_start + timedelta(days=i)
        chart_data["labels"].append(str(date.strftime("%m/%d")))
        
        if i < 7:  # Historical actual data
            day_sales = sales_df[sales_df['Date'].dt.date == date.date()]
            total_day_sales = day_sales['Quantity'].sum() if not day_sales.empty else 0
            chart_data["actual"].append(int(total_day_sales))
            chart_data["forecast"].append(None)
            
            # For historical period, we need to calculate SMA/EMA from the data
            # This is a simplified approach - in practice, you'd want to calculate these properly
            chart_data["sma"].append(None)  # Historical SMA would need proper calculation
            chart_data["ema"].append(None)  # Historical EMA would need proper calculation
        else:  # Future predictions
            # Aggregate predictions for this day across all products
            day_index = i - 7  # 0-6 for the 7 prediction days
            total_predicted = 0.0
            total_sma = 0.0
            total_ema = 0.0
            
            for product_name, forecast_info in all_forecasts.items():
                # Estimate daily prediction from total forecasted quantity
                daily_pred = forecast_info['forecasted_quantity'] / 7
                total_predicted += daily_pred
                
                # Get SMA and EMA for this day
                sma_data = forecast_info.get('sma', [])
                ema_data = forecast_info.get('ema', [])
                
                if day_index < len(sma_data):
                    total_sma += sma_data[day_index]
                if day_index < len(ema_data):
                    total_ema += ema_data[day_index]
            
            chart_data["actual"].append(None)
            chart_data["forecast"].append(float(round(total_predicted, 1)))
            chart_data["sma"].append(float(round(total_sma, 1)))
            chart_data["ema"].append(float(round(total_ema, 1)))
    
    return {
        'products': forecast_results,
        'chart_data': chart_data,
        'timeframe_info': {
            'start_date': str(start_dt.strftime("%Y-%m-%d")),
            'end_date': str(end_dt.strftime("%Y-%m-%d")),
            'days_count': int(days_count)
        },
        'model_summary': {
            'total_products': len(forecast_results),
            'ml_models': len([p for p in forecast_results if p['model_type'] == 'random_forest']),
            'ema_models': len([p for p in forecast_results if p['model_type'] in ['ema', 'ema_fallback']]),
            'naive_models': len([p for p in forecast_results if p['model_type'] in ['naive', 'naive_fallback']])
        }
    }


@app.route("/api/inventory/refresh")
def api_inventory_refresh():
    """API endpoint to refresh inventory data and return updated information"""
    if not _require_admin():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        # Refresh inventory suggestions
        ml_demand.inventory_advisor()
        
        # Get updated inventory data
        products_df = _load_csv(PRODUCTS_CSV)
        sales_df = _load_csv(SALES_CSV)
        reorder_df = _load_csv(REORDER_CSV)
        
        # Calculate recent sales (last 7 days)
        if not sales_df.empty and 'Date' in sales_df.columns:
            sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
            cutoff_date = datetime.now() - timedelta(days=7)
            recent_sales = sales_df[sales_df['Date'] >= cutoff_date]
            
            if not recent_sales.empty:
                sales_by_item = recent_sales.groupby('Item')['Quantity'].sum().to_dict()
            else:
                sales_by_item = {}
        else:
            sales_by_item = {}
        
        # Create unified inventory data
        unified_inventory = []
        products = products_df.to_dict(orient="records")
        
        for product in products:
            product_name = product['name']
            current_stock = int(product['stock'])
            recent_sales_7d = sales_by_item.get(product_name, 0)
            
            # Get predicted demand from reorder suggestions
            predicted_demand = 0.0
            reorder_suggestion = reorder_df[reorder_df['Item'] == product_name]
            if not reorder_suggestion.empty:
                predicted_demand = float(reorder_suggestion.iloc[0]['Predicted_Demand'])
            
            # Calculate dynamic reorder flag
            reorder_flag = _calculate_reorder_flag(current_stock, predicted_demand)
            
            # Determine status
            if current_stock <= 0:
                status = "Out of Stock"
            elif current_stock <= 5:
                status = "Low Stock"
            else:
                status = "In Stock"
            
            unified_inventory.append({
                'product': product_name,
                'current_stock': current_stock,
                'recent_sales_7d': recent_sales_7d,
                'status': status,
                'predicted_demand_7d': round(predicted_demand, 1),
                'reorder_flag': reorder_flag
            })
        
        return jsonify({
            "status": "success",
            "inventory_data": unified_inventory,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/refresh")
def debug_refresh():
    if not _require_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))
    
    # Reload CSVs into memory
    try:
        products_df = _load_csv(PRODUCTS_CSV)
        sales_df = _load_csv(SALES_CSV)
        users_df = _load_csv(USERS_CSV)
        
        return jsonify({
            "status": "OK",
            "products_count": len(products_df),
            "sales_count": len(sales_df),
            "users_count": len(users_df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    with app.app_context():
        init_app_files()
    app.run(debug=True)


