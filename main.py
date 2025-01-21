Creating a full-fledged "smart-stock-optimizer" using machine learning can be quite extensive, but I can provide you with a straightforward Python script that outlines the basic structure. We'll use historical stock data to train a simple machine learning model for predicting future prices and optimize a portfolio based on predictions. We'll use libraries like `pandas`, `numpy`, `scikit-learn`, and `yfinance`.

Please note that this is just a starting point, and refining the model for real-world usage involves a lot more complexity including feature selection, hyperparameter tuning, model evaluation, risk assessment, and much more.

```python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Exception handling for missing packages
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError(f"Missing package: {e.name}. Please install it by running `pip install {e.name}` before executing the script.")

# Fetch historical stock data
def fetch_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        data = data.dropna()  # Handle missing data by dropping any rows with NaN values
    except Exception as e:
        raise Exception(f"Failed to fetch data: {e}")
    return data

# Train a simple model to predict future stock prices
def train_model(data):
    try:
        X = data[:-1]  # Features (all but last row)
        y = data[1:]   # Labels (all but first row)

        # Split data into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize a basic Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Fit the model
        model.fit(X_train, y_train)

        # Predict on test set and calculate error
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        print(f"Model training completed. MSE on test data: {mse:.4f}")
    except Exception as e:
        raise Exception(f"Model training failed: {e}")

    return model

# Optimization function to minimize risk in the portfolio
def optimize_portfolio(model, data):
    try:
        predictions = model.predict(data[-1:])[0]

        # Objective function to minimize (negative sharpe ratio)
        def objective(weights):
            returns = np.dot(weights, predictions)
            # Assume risk as standard deviation of the portfolio
            risk = np.sqrt(np.dot(weights.T, np.dot(data.cov(), weights)))
            return -returns / risk  # Negative Sharpe Ratio

        # Constraints: Sum of weights = 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        # Bounds for each weight (between 0 and 1)
        bounds = tuple((0, 1) for _ in range(len(data.columns)))
        # Initial guess for the portfolio weights
        initial_weights = np.array([1.0 / len(data.columns)] * len(data.columns))

        # Run the optimization
        optimal = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        # Ensure optimization was successful
        if not optimal.success:
            raise Exception("Optimization failed")

        print(f"Optimal allocation: {optimal.x}")
        print(f"Expected portfolio return: {np.dot(optimal.x, predictions)}")
    except Exception as e:
        raise Exception(f"Portfolio optimization failed: {e}")

# Main execution function
def run_optimizer():
    tickers = ['AAPL', 'GOOGL', 'MSFT']  # Example stock tickers
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    try:
        data = fetch_data(tickers, start_date, end_date)
        model = train_model(data)
        optimize_portfolio(model, data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_optimizer()
```

### Key Points:
- **Data Fetching**: We use the `yfinance` library to fetch historical prices.
- **Model Training**: Uses Random Forest to create a simple prediction model. Other models like LSTM, ARIMA could be explored.
- **Error Handling**: Implements basic exception handling.
- **Portfolio Optimization**: Uses a scipy optimization function to minimize risk based on predicted returns.
- **Comments**: Each part of the code is commented to help understand its function.

This is a simplified implementation and, in practice, portfolio optimization is a much more nuanced topic involving sophisticated models and finance domain knowledge.