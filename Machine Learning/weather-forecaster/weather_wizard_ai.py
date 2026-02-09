# THE MODEL IS NOT LEARNING AND CURRENTLY PREDICTS THE SAME VALUE FOR EVERY DAY.
# THIS IS A MAJOR ISSUE THAT NEEDS TO BE FIXED.

# TODO
# 1. Figure out why they model doesnt seem to be learning
# 2. Work on making the predictions more accurate
# 3. Working on UX elements and quality of life features (date inputs, tuning hyperparameters, etc.)
# 4. Add a feature to the model that allows it to predict the weather for the next 7 days

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

today = datetime.date.today()


# 1. Data Collection
def collect_data():
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": 52.0417,
        "longitude": -0.7558,
        "start_date": "2022-08-01",  # date format: YYYY-MM-DD
        "timezone": "GMT",
        "end_date": "2024-04-30",
        "daily": ["apparent_temperature_max", "apparent_temperature_min"]
    }
    response = requests.get(url, params=params)
    data = response.json()['daily']
    df = pd.DataFrame({
        'date': data['time'],
        'apparent_temperature_max': data['apparent_temperature_max'],
        'apparent_temperature_min': data['apparent_temperature_min']
    })
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    return df



# 2. Data Preprocessing (df = dataframe)
def preprocess_data(df):
    # One-hot encoding of the month
    df = pd.get_dummies(df, columns=['month'])

    # Select numeric columns only
    df_numeric = df.select_dtypes(include=[np.number]) # Select all numeric columns

    # Fill missing values
    df_numeric = df_numeric.fillna(df_numeric.mean())

    # Convert to numpy arrays and scale to [0, 1]
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    output_scaler = MinMaxScaler(feature_range=(0, 1))

    # The output data is the first column (temperature)
    output_data = df_numeric.iloc[:, 0].values.reshape(-1, 1)
    scaled_output_data = output_scaler.fit_transform(output_data)

    # The input data is all columns
    scaled_input_data = input_scaler.fit_transform(df_numeric.values)

    # Concatenate the scaled input and output data
    scaled_data = np.concatenate([scaled_output_data, scaled_input_data], axis=1)

    return scaled_data, input_scaler, output_scaler



# 3. Feature Selection/Engineering
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 4. Model Training
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (hn, cn) = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(X_train, y_train, num_epochs, input_size, hidden_size, num_layers, seq_length, learning_rate):
    model = LSTM(num_classes=1, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, seq_length=seq_length)
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model.forward(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Print loss for every epoch instead of every 100 epochs
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    return model



# 5. Model Evaluation
def evaluate_model(model, X_test, y_test, input_scaler, output_scaler, train_size):
    model.eval()
    train_predict = model(X_test)
    data_predict = train_predict.data.numpy()
    dataY_plot = y_test.data.numpy()

    data_predict = data_predict.reshape(-1, 1)
    dataY_plot = dataY_plot.reshape(-1, 1)
    data_predict = output_scaler.inverse_transform(data_predict)
    dataY_plot = output_scaler.inverse_transform(dataY_plot)

    plt.axvline(x=train_size, c='r', linestyle='--')
    plt.plot(dataY_plot)
    plt.plot(data_predict)
    plt.suptitle('Time-Series Prediction')
    plt.show()



# Prediction
def make_predictions(model, X, output_scaler):
    model.eval()
    # Make predictions on input data
    data_predict = model(X)
    data_predict = data_predict.data.numpy()
    data_predict = output_scaler.inverse_transform(data_predict)

    seq_length = 4
    n_features = 3  # This should match the input_size of your LSTM model

    # Get the most recent sequence of observations
    next_day = X[-1:, :, :]  # Take the last sequence from X

    # Predict next day
    next_day_pred = model(next_day.to(device))
    next_day_pred = next_day_pred.detach().numpy()
    next_day_pred = output_scaler.inverse_transform(next_day_pred)

    #print today's temperature
    todays_temperature = next_day_pred[0, 0]
    print("Today's temperature is {:.1f} degrees.".format(todays_temperature))

    # Convert predicted temperature to weather forecast
    temp = next_day_pred[0, 0]
    if temp < 0:
        forecast = "It's going to be freezing tomorrow! The temperature is expected to be {:.1f} degrees.".format(temp)
    elif temp < 10:
        forecast = "It's going to be quite cold tomorrow with a predicted temperature of {:.1f} degrees.".format(temp)
    elif temp < 20:
        forecast = "The weather will be mild tomorrow with a temperature around {:.1f} degrees.".format(temp)
    elif temp < 30:
        forecast = "It will be warm tomorrow! The temperature is expected to be around {:.1f} degrees.".format(temp)
    else:
        forecast = "It's going to be hot tomorrow with an expected temperature of {:.1f} degrees.".format(temp)

    return forecast





def main():
    seq_length = 100  # number of previous time steps to use as input variables to predict the next time period
    num_epochs = 250
    learning_rate = 0.001
    input_size = 3
    hidden_size = 64
    num_layers = 2

    # Data collection
    df = collect_data()

    # Print the dataframe
    print(df.head())
    print(df.describe())

    # Data preprocessing
    data, input_scaler, output_scaler = preprocess_data(df)

    # Feature selection/engineering
    X, y = create_sequences(data, seq_length)

    # Check if sequences of the correct length were generated
    if len(X) == 0 or len(y) == 0:
        raise ValueError(
            "The date range is too short to create sequences of the specified length. Please choose a longer date range.")

    # Continue with the rest of the code...

    # Splitting data into training and testing sets
    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size
    X_train = torch.Tensor(np.array(X[0:train_size]))
    y_train = torch.Tensor(np.array(y[0:train_size]))
    X_test = torch.Tensor(np.array(X[train_size:len(X)]))
    y_test = torch.Tensor(np.array(y[train_size:len(y)]))

    # Model training
    model = train_model(X_train, y_train, num_epochs, input_size, hidden_size, num_layers, seq_length, learning_rate)

    # Model evaluation
    evaluate_model(model, X_test, y_test, input_scaler, output_scaler, train_size)

    # Making predictions
    predictions = make_predictions(model, X_test, output_scaler)
    print(predictions)


if __name__ == "__main__":
    main()

