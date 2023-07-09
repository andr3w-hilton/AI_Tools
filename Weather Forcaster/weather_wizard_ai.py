import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


# 1. Data Collection
def collect_data():
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "start_date": "2021-01-01",
        "end_date": "2021-12-31",
        "hourly": "temperature_2m"
    }
    response = requests.get(url, params=params)
    data = response.json()['hourly']
    df = pd.DataFrame(data)
    return df


# 2. Data Preprocessing (df = dataframe)
def preprocess_data(df):
    # Select numeric columns only
    df_numeric = df.select_dtypes(include=[np.number])

    # Fill missing values
    df_numeric = df_numeric.fillna(df_numeric.mean())

    # Convert to numpy arrays and scale to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_numeric.values)

    return scaled_data, scaler


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

        if epoch % 100 == 0:
          print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    return model


# 5. Model Evaluation
def evaluate_model(model, X_test, y_test):
    model.eval()
    train_predict = model(X_test)
    data_predict = train_predict.data.numpy()
    dataY_plot = y_test.data.numpy()

    data_predict = scaler.inverse_transform(data_predict)
    dataY_plot = scaler.inverse_transform(dataY_plot)

    plt.axvline(x=train_size, c='r', linestyle='--')
    plt.plot(dataY_plot)
    plt.plot(data_predict)
    plt.suptitle('Time-Series Prediction')
    plt.show()


# 6. Prediction
def make_predictions(model, X):
    model.eval()
    data_predict = model(X)
    data_predict = data_predict.data.numpy()
    data_predict = scaler.inverse_transform(data_predict)
    return data_predict


def main():
    seq_length = 4  # number of previous time steps to use as input variables to predict the next time period
    num_epochs = 2000
    learning_rate = 0.01
    input_size = 1
    hidden_size = 2
    num_layers = 1

    # Data collection
    df = collect_data()

    # Data preprocessing
    data, scaler = preprocess_data(df)

    # Feature selection/engineering
    X, y = create_sequences(data, seq_length)

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
    evaluate_model(model, X_test, y_test)

    # Making predictions
    predictions = make_predictions(model, X_test)


if __name__ == "__main__":
    main()
