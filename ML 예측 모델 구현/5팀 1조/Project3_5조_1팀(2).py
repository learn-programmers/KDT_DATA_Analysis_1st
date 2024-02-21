import argparse
import pandas as pd
import matplotlib.pyplot as plt
from FinanceDataReader import DataReader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prophet import Prophet

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--start_date', type=str, default='2007-01-01', help='Start date for data')
parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date for data')
args = parser.parse_args()

# Load Samsung stock price data
samsung_data = DataReader('005930', args.start_date, args.end_date)

# Prepare the data for modeling
samsung_data['Close'] = samsung_data['Close'].pct_change()
samsung_data.dropna(inplace=True)

# Split data into training and test sets
train_data = samsung_data[samsung_data.index < '2023-01-01']
test_data = samsung_data[samsung_data.index >= '2023-01-01']

# Prepare the data for Prophet
df_prophet = train_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

# Models
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Prophet': Prophet(daily_seasonality=False, yearly_seasonality=True)
}

# Dictionary to store predictions
predictions_dict = {}

# Train and forecast with each model
for name, model in models.items():
    if name != 'Prophet':
        # Prepare the input features
        X_train = train_data.drop(['Close'], axis=1)
        y_train = train_data['Close']
        X_test = test_data.drop(['Close'], axis=1)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        predictions_dict[name] = predictions
        
        # Evaluate predictions
        mse = mean_squared_error(test_data['Close'], predictions)
        r2 = r2_score(test_data['Close'], predictions)
        mae = mean_absolute_error(test_data['Close'], predictions)
        print(f'{name} MSE: {mse}, R2: {r2}, MAE: {mae}')
    else:
        # Train the Prophet model
        model.fit(df_prophet)
        
        # Make future dataframe for 2023
        future = model.make_future_dataframe(periods=245)
        
        # Predict for 2023
        forecast = model.predict(future)
        predictions_dict[name] = forecast['yhat'][-245:].values
        
        # Evaluate predictions
        mse = mean_squared_error(test_data['Close'], predictions_dict[name])
        r2 = r2_score(test_data['Close'], predictions_dict[name])
        mae = mean_absolute_error(test_data['Close'], predictions_dict[name])
        print(f'{name} MSE: {mse}, R2: {r2}, MAE: {mae}')

# Plot the actual vs predicted values for 2023
for name, predictions in predictions_dict.items():
    plt.figure(figsize=(10, 5))
    plt.plot(test_data.index, test_data['Close'], label='Actual', color='blue')
    plt.plot(test_data.index, predictions, label='Predicted', color='red')
    plt.title(f'{name} Actual vs Predicted for 2023')
    plt.xlabel('Date')
    plt.ylabel('Percent Change')
    plt.legend()
    plt.show()
