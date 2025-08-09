# Import required libraries
import tkinter as tk
from tkinter import ttk, messagebox
from binance.client import Client
import threading
import time

# Initialize the client
client = Client()

# List of popular cryptocurrencies
POPULAR_CRYPTOS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
    'DOGEUSDT', 'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT'
]

class CryptoTraderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CryptoTrader")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a modern theme
        
        # Configure colors
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Price.TLabel', font=('Arial', 14))
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_label = ttk.Label(self.main_frame, text="CryptoTrader Dashboard", style='Header.TLabel')
        header_label.pack(pady=(0, 20))
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.prices_tab = ttk.Frame(self.notebook, padding=10)
        self.alert_tab = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.prices_tab, text="Live Prices")
        self.notebook.add(self.alert_tab, text="Price Alerts")
        
        # Setup the prices tab
        self.setup_prices_tab()
        
        # Setup the alert tab
        self.setup_alert_tab()
        
        # Initialize variables
        self.update_thread = None
        self.alert_thread = None
        self.stop_thread = False
        self.alert_active = False
        
        # Start updating prices
        self.start_price_updates()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_prices_tab(self):
        # Create a frame for the controls
        controls_frame = ttk.Frame(self.prices_tab)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add refresh button
        refresh_btn = ttk.Button(controls_frame, text="Refresh Now", command=self.refresh_prices)
        refresh_btn.pack(side=tk.RIGHT, padx=5)
        
        # Create a frame for the prices
        prices_frame = ttk.Frame(self.prices_tab)
        prices_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for prices
        columns = ("symbol", "price", "change_24h")
        self.price_tree = ttk.Treeview(prices_frame, columns=columns, show="headings")
        
        # Define headings
        self.price_tree.heading("symbol", text="Symbol")
        self.price_tree.heading("price", text="Price (USDT)")
        self.price_tree.heading("change_24h", text="24h Change")
        
        # Define columns
        self.price_tree.column("symbol", width=100)
        self.price_tree.column("price", width=150)
        self.price_tree.column("change_24h", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(prices_frame, orient=tk.VERTICAL, command=self.price_tree.yview)
        self.price_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.price_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status label
        self.status_label = ttk.Label(self.prices_tab, text="Ready")
        self.status_label.pack(pady=(10, 0), anchor=tk.W)
    
    def setup_alert_tab(self):
        # Create frames
        form_frame = ttk.Frame(self.alert_tab)
        form_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Symbol selection
        ttk.Label(form_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        symbol_combo = ttk.Combobox(form_frame, textvariable=self.symbol_var, values=POPULAR_CRYPTOS)
        symbol_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Target price
        ttk.Label(form_frame, text="Target Price (USDT):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.target_price_var = tk.StringVar(value="115000")
        target_entry = ttk.Entry(form_frame, textvariable=self.target_price_var, width=15)
        target_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Alert type
        ttk.Label(form_frame, text="Alert Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.alert_type_var = tk.StringVar(value="above")
        alert_type_frame = ttk.Frame(form_frame)
        alert_type_frame.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        
        ttk.Radiobutton(alert_type_frame, text="Above", variable=self.alert_type_var, value="above").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(alert_type_frame, text="Below", variable=self.alert_type_var, value="below").pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.start_alert_btn = ttk.Button(button_frame, text="Start Alert", command=self.toggle_alert)
        self.start_alert_btn.pack(side=tk.LEFT, padx=5)
        
        # Current price display
        self.current_price_frame = ttk.LabelFrame(self.alert_tab, text="Current Price")
        self.current_price_frame.pack(fill=tk.X, pady=10)
        
        self.current_price_label = ttk.Label(self.current_price_frame, text="-", style="Price.TLabel")
        self.current_price_label.pack(pady=10)
        
        # Alert status
        self.alert_status_frame = ttk.LabelFrame(self.alert_tab, text="Alert Status")
        self.alert_status_frame.pack(fill=tk.X, pady=10)
        
        self.alert_status_label = ttk.Label(self.alert_status_frame, text="No alert active", style="Price.TLabel")
        self.alert_status_label.pack(pady=10)
    
    def get_crypto_price(self, symbol):
        """Get the current price for a given cryptocurrency symbol"""
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Error getting price: {e}")
            return None
    
    def get_24h_change(self, symbol):
        """Get 24-hour price change for a symbol"""
        try:
            ticker_24h = client.get_ticker(symbol=symbol)
            change_percent = float(ticker_24h['priceChangePercent'])
            return change_percent
        except Exception as e:
            print(f"Error getting 24h change: {e}")
            return None
    
    def refresh_prices(self):
        """Refresh the prices in the treeview"""
        self.status_label.config(text="Refreshing prices...")
        
        # Clear the treeview
        for item in self.price_tree.get_children():
            self.price_tree.delete(item)
        
        # Add prices for popular cryptos
        for symbol in POPULAR_CRYPTOS:
            price = self.get_crypto_price(symbol)
            change = self.get_24h_change(symbol)
            
            if price is not None:
                # Format the price
                price_str = f"${price:,.2f}"
                
                # Format the change
                if change is not None:
                    if change > 0:
                        change_str = f"+{change:.2f}%"
                        tags = ("positive",)
                    elif change < 0:
                        change_str = f"{change:.2f}%"
                        tags = ("negative",)
                    else:
                        change_str = f"{change:.2f}%"
                        tags = ()
                else:
                    change_str = "N/A"
                    tags = ()
                
                # Insert into treeview
                self.price_tree.insert("", tk.END, values=(symbol, price_str, change_str), tags=tags)
        
        # Configure tag colors
        self.price_tree.tag_configure("positive", foreground="green")
        self.price_tree.tag_configure("negative", foreground="red")
        
        # Update status
        self.status_label.config(text=f"Last updated: {time.strftime('%H:%M:%S')}")
    
    def start_price_updates(self):
        """Start the thread for updating prices"""
        self.stop_thread = False
        self.update_thread = threading.Thread(target=self.update_prices_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def update_prices_thread(self):
        """Thread function to update prices periodically"""
        while not self.stop_thread:
            # Update prices on the GUI thread
            self.root.after(0, self.refresh_prices)
            
            # Sleep for 30 seconds
            for _ in range(30):
                if self.stop_thread:
                    break
                time.sleep(1)
    
    def toggle_alert(self):
        """Toggle the price alert on/off"""
        if not self.alert_active:
            # Start the alert
            try:
                symbol = self.symbol_var.get()
                target_price = float(self.target_price_var.get())
                alert_type = self.alert_type_var.get()
                
                # Update UI
                self.alert_active = True
                self.start_alert_btn.config(text="Stop Alert")
                self.alert_status_label.config(text=f"Monitoring {symbol} for {alert_type} ${target_price:,.2f}")
                
                # Start the alert thread
                self.stop_thread = False
                self.alert_thread = threading.Thread(target=self.price_alert_thread, 
                                                   args=(symbol, target_price, alert_type))
                self.alert_thread.daemon = True
                self.alert_thread.start()
                
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid target price.")
        else:
            # Stop the alert
            self.alert_active = False
            self.start_alert_btn.config(text="Start Alert")
            self.alert_status_label.config(text="No alert active")
    
    def price_alert_thread(self, symbol, target_price, alert_type):
        """Thread function to monitor price for alert"""
        while self.alert_active and not self.stop_thread:
            # Get current price
            current_price = self.get_crypto_price(symbol)
            
            if current_price is not None:
                # Update the price display
                self.root.after(0, lambda p=current_price: 
                               self.current_price_label.config(text=f"${p:,.2f}"))
                
                # Check if alert condition is met
                alert_triggered = False
                if alert_type == "above" and current_price >= target_price:
                    alert_triggered = True
                elif alert_type == "below" and current_price <= target_price:
                    alert_triggered = True
                
                if alert_triggered:
                    # Show alert on the GUI thread
                    self.root.after(0, lambda: self.show_alert(symbol, current_price, target_price, alert_type))
                    break
            
            # Sleep for 5 seconds
            for _ in range(5):
                if not self.alert_active or self.stop_thread:
                    break
                time.sleep(1)
    
    def show_alert(self, symbol, current_price, target_price, alert_type):
        """Show the alert message"""
        # Play alert sound
        self.root.bell()
        
        # Show message box
        messagebox.showinfo("Price Alert", 
                           f"🚨 ALERT! 🚨\n\n{symbol} price is now ${current_price:,.2f}\n" 
                           f"Target of ${target_price:,.2f} ({alert_type}) has been reached!")
        
        # Reset the alert
        self.alert_active = False
        self.start_alert_btn.config(text="Start Alert")
        self.alert_status_label.config(text="Alert triggered! Set a new alert.")
    
    def on_close(self):
        """Handle window close event"""
        self.stop_thread = True
        self.root.destroy()

# Main function
def main():
    root = tk.Tk()
    app = CryptoTraderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()