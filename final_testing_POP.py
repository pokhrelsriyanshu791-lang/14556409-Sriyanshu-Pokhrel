import csv
from datetime import datetime
import time
import random
import os

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
1
# File location details for file handling
STOCKS_FILE = os.path.join(SCRIPT_DIR, "stocks.csv")
SALES_FILE = os.path.join(SCRIPT_DIR, "sales.csv")
USERS_FILE = os.path.join(SCRIPT_DIR, "users.csv")

ADMIN_USERNAME = "admin123"
ADMIN_PASSWORD = "Ghimire@123"
 
# Base class according to requirement
class StockItem:
    category = "Car accessories"

    def __init__(self, stock_code, quantity, price):
        self.__stock_code = stock_code
        self.__quantity = quantity
        self.__price = price

    def set_price(self, price):
        self.__price = price

    def get_price(self):
        return self.__price

    def get_price_with_vat(self):
        return self.__price * 1.175

    def set_quantity(self, quantity):
        self.__quantity = quantity

    def get_quantity(self):
        return self.__quantity

    def increase_stock(self, amount):
        if amount < 1 or self.__quantity + amount > 100:
            print("Invalid stock increase amount.")
        else:
            self.__quantity += amount

    def sell_stock(self, amount):
        if amount < 1:
            print("Invalid sale amount.")
            return False
        elif amount > self.__quantity:
            print("Not enough stock available.")
            return False
        else:
            self.__quantity -= amount
            return True

    def get_stock_name(self):
        return "Unknown Stock Name"

    def get_stock_description(self):
        return "Unknown Stock Description"

    def __str__(self):
        return (f"Stock Code: {self.__stock_code}, Name: {self.get_stock_name()}, "
                f"Description: {self.get_stock_description()}, Quantity: {self.__quantity}, "
                f"Price Before VAT: Rs.{self.__price}, Price After VAT: Rs.{self.get_price_with_vat()}")

# Deerived Nav class for navigation system
class NavSys(StockItem):
    def __init__(self, stock_code, quantity, price, brand):
        super().__init__(stock_code, quantity, price)
        self.__brand = brand

    def get_stock_name(self):
        return "Navigation system"

    def get_stock_description(self):
        return "GeoVision Sat Nav"

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}, Brand: {self.__brand}"

# Main Menu function to display the first output
def main_menu():
    print("\n----Ghimire Automobiles----\n")
    print("1. Admin Login")
    print("2. User Login")
    print("3. Exit")
    choice = input("Choose an option: ")

# Conditions to direct user to corresponding functions
    if choice == "1":
        admin_login()
    elif choice == "2":
        user_login()
    elif choice == "3":
        print("Thank you for visiting Ghimire Automobiles.")
        exit()
    else:
        print("Invalid choice. Try again.")
        main_menu()

# Function to log into admin panel
def admin_login():
    attempts = 0
    while attempts < 3:
        username = input("Enter username for admin login: ")
        password = input("Enter password for admin login: ")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            print("\n----Welcome to Admin Section----")
            admin_menu()
            return
        else:
            print("Mismatched Credentials")
            attempts += 1
            if attempts == 3:
                print("Too many wrong inputs, Admin section has been locked for a minute")
                time.sleep(60)
                main_menu()

# Functions for options inside admin panel
def admin_menu():
    while True:
        print("1. Manage Stocks")
        print("2. Check Sales")
        print("3. Manage Pricings")
        print("4. Logout")
        choice = input("Choose an option: ")

# Conditions to direct user to corresponding functions for each item
        if choice == "1":
            manage_stocks()
        elif choice == "2":
            check_sales()
        elif choice == "3":
            manage_pricings()
        elif choice == "4":
            main_menu()
            return
        else:
            print("Invalid choice. Try again.")

# Function to manage stocks directly into the stocks.csv file
def manage_stocks():
    print("\n----Manage Stocks----")
    try:
        with open(STOCKS_FILE, "r") as file: 
            reader = csv.DictReader(file)
            stocks = list(reader)

        for idx, stock in enumerate(stocks, start=1):
            print(f"{idx}. {stock['Category']} - {stock['Item']} ({stock['Stock']} in stock)")

        choice = int(input("Select an item to modify stock (0 to go back): "))
        if choice == 0:
            return

        selected_stock = stocks[choice - 1]
        print(f"Selected: {selected_stock['Category']} - {selected_stock['Item']}")
        change = int(input("Enter quantity to add/remove (use negative for removal): "))
        updated_stock = max(0, int(selected_stock['Stock']) + change)
        selected_stock['Stock'] = str(updated_stock)

        with open(STOCKS_FILE, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=stocks[0].keys())
            writer.writeheader()
            writer.writerows(stocks)

        print("Stock updated successfully.")
    except (ValueError, IndexError):
        print("Invalid input. Returning to menu.")
    except FileNotFoundError:
        print("Corresponding file not found!")

# Function to check sales from sales.csv file
def check_sales():
    print("\n----Sales Data----")
    try:
        with open(SALES_FILE, "r") as file:
            reader = csv.DictReader(file)
            sales = list(reader)

        if not sales:
            print("No sales data available.")
            return

        total_sales = 0
        for sale in sales:
            print(f"Username: {sale['Username']}, Item: {sale['Item']}, Quantity: {sale['Quantity']}, Total Price: Rs.{sale['Total Price']}, Date: {sale['Date']}")
            total_sales += float(sale['Total Price'])

        print(f"\nTotal Sales: Rs.{total_sales}")
    except FileNotFoundError:
        print("No corresponding File found!")

# Function to change prices in the stocks.csv file to update in the user section
def manage_pricings():
    print("\n----Manage Pricings----")
    try:
        with open(STOCKS_FILE, "r") as file:
            reader = csv.DictReader(file)
            stocks = list(reader)

        for idx, stock in enumerate(stocks, start=1):
            print(f"{idx}. {stock['Category']} - {stock['Item']} (Current Price: Rs.{stock['Price']})")

        choice = int(input("\nSelect an item to update its price (0 to go back): "))
        if choice == 0:
            return

        selected_stock = stocks[choice - 1]
        new_price = float(input(f"Enter new price for {selected_stock['Item']}: Rs."))
        selected_stock['Price'] = str(new_price)

        with open(STOCKS_FILE, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=stocks[0].keys())
            writer.writeheader()
            writer.writerows(stocks)

        print("Prices have been updated sucessfully!")
    except (ValueError, IndexError):
        print("Invalid input. Please try again!")
    except FileNotFoundError:
        print("Stocks file not found.")
        
# Function to enter user id and password and check if it exists in users.csv file
def user_login():
    username = input("Enter your User ID: ")
    password = input("Enter your password: ")

    try:
        with open(USERS_FILE, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["Username"] == username and row["Password"] == password:
                    print("Login successful!")
                    user_menu(username)
                    return

        print("Invalid credentials. Would you like to register? (yes/no)")
        if input().lower() == "yes":
            register_user()
        else:
            print("Thank you for visiting Ghimire Automobiles.")
    except FileNotFoundError:
        print("Users file not found.")

# Function if user's login credentials are wrong and write new user data to users.csv file 
def register_user():
    print("\nRegister new user")
    username = input("Enter a 5-digit user ID: ")
    while len(username) != 5 or not username.isdigit():
        print("Username must be of 5 digits!!")
        username = input("Enter a 5-digit username: ")

    password = input("Enter a password (max 10 characters): ")
    while len(password) > 10:
        print("Password cannot be more than 10 characters!!")
        password = input("Enter your password(Max 10 characters!)")

    try:
        with open(USERS_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([username, password])

        print("Registration successful! Please login.")
        user_login()
    except FileNotFoundError:
        print("Users file not found.")

# Function for displaying the categories in user menu
def user_menu(username):
    cart = []  # To hold the cart until user clicks no to adding items
    while True:
        print("\n1. Tires")
        print("2. Engine Oils")
        print("3. Brakes")
        print("4. Engine Parts")
        print("5. Lights")
        print("6. View Cart and Checkout")
        print("7. Quit")
        choice = input("Choose a option: ")

        if choice == "7":
            print("Thank you for visiting Ghimire Automobiles.")
            break
        elif choice == "6":
            if cart:
                generate_bill(cart, username)
            else:
                print("Your cart is empty!")
        elif choice in ["1", "2", "3", "4", "5"]:
            category_map = {
                "1": "Tires",
                "2": "Engine Oils",
                "3": "Brakes",
                "4": "Engine Parts",
                "5": "Lights"
            }
            specific_menu(category_map[choice], username, cart)
        else:
            print("Invalid choice. Please try again!")

# Function to display every item in the categories and the items to be extracted from stocks.csv file
def specific_menu(category, username, cart):
    try:
        with open(STOCKS_FILE, "r") as file:
            reader = csv.DictReader(file)
            items = [(row["Item"], float(row["Price"]), int(row["Stock"])) for row in reader if row["Category"] == category]

        print(f"\n----{category} Menu----")
        for idx, (item, price, stock) in enumerate(items, start=1):
            print(f"{idx}. {item} - Rs.{price} ({stock} in stock)")

        choice = input("Choose an option (0 to go back)")
        if choice == "0":
            return

        try:
            selected_item, price, stock = items[int(choice) - 1]
            print(f"Available stock for {selected_item}: {stock}")
            quantity = int(input(f"Enter quantity for {selected_item}: "))
            if quantity > stock:
                print("Not enough stock available.")
            else:
                add_to_cart(cart, category, selected_item, price, quantity)
        except (IndexError, ValueError):
            print("Invalid choice. Please try again!")
            specific_menu(category, username, cart)
    except FileNotFoundError:
        print("Stock file does not exist!")

# Function to enable user to add items to his cart
def add_to_cart(cart, category, item_name, price, quantity):
    cart.append({"Category": category, "Item": item_name, "Price": price, "Quantity": quantity})
    print(f"{quantity} x {item_name} added to cart.")

# Function to generate bill by checking OTP and bill with and without VAT
def generate_bill(cart, username):
    if not cart:
        print("Your have nothing in your cart!")
        return

    # 4-digit OTP generation
    otp = random.randint(1000, 9999)
    print(f"Your OTP is: {otp}")

    entered_otp = input("Enter your OTP to validate checkout: ")

    # OTP validation for bill checking out
    if str(otp) != entered_otp:
        print("Invalid OTP. Cannot checkout and generate bill")
        return

    total_price = 0
    vat_rate = 0.175 
    print("\n----Final Bill----")
    print(f"Customer: {username}")
    print("Item\t\tQuantity\tPrice\tTotal")

    for item in cart:
        item_total = item["Price"] * item["Quantity"]
        total_price += item_total
        print(f"{item['Item']}\t{item['Quantity']}\t\t{item['Price']}\t{item_total}")
        update_stock(item["Category"], item["Item"], item["Quantity"])
        record_sale(username, item["Category"], item["Item"], item["Quantity"], item_total)

    vat_amount = total_price * vat_rate
    grand_total = total_price + vat_amount

    print(f"\nTotal without VAT: Rs.{total_price}")
    print(f"VAT (17.5%): Rs.{vat_amount}")
    print(f"Grand Total: Rs.{grand_total}")
    print("Thank you for choosing Ghimire Automobiles :)")
    cart.clear()  # Empty user's cart after bill has been given

# Function to update the item stocks after user purchase
def update_stock(category, item_name, quantity_purchased):
    with open(STOCKS_FILE, "r") as stocks_file:
        stocks = list(csv.DictReader(stocks_file))

    for stock in stocks:
        if stock["Category"] == category and stock["Item"] == item_name:
            stock["Stock"] = str(int(stock["Stock"]) - quantity_purchased)

    with open(STOCKS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Category", "Item", "Stock", "Price"])
        writer.writeheader()
        writer.writerows(stocks)

# Function to keep track of sales of each user
def record_sale(username, category, item_name, quantity, total_price):
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sale_entry = {
        "Username": username,
        "Category": category,
        "Item": item_name,
        "Quantity": quantity,
        "Total Price": total_price,
        "Date": current_date
    }

    with open(SALES_FILE, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Username", "Category", "Item", "Quantity", "Total Price", "Date"])
        if file.tell() == 0:  
            writer.writeheader()
        writer.writerow(sale_entry)

if __name__ == "__main__":
    main_menu()
