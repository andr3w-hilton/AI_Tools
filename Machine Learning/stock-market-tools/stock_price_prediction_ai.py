# Import necessary libraries
import math
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


# Define color codes for terminal output
class colors:
    GREEN = '\033[32m'  # GREEN
    YELLOW = '\033[33m'  # YELLOW
    RED = '\033[31m'  # RED
    RESET = '\033[0m'  # RESET color


# Define background color codes for terminal output
class bkgrn_colors:
    GREEN = '\033[42m'  # GREEN
    YELLOW = '\033[43m'  # YELLOW
    RED = '\033[41m'  # RED
    RESET = '\033[0m'  # RESET color


# Print a fancy banner
print("""                                                             
 █████╗ ██╗    ███████╗████████╗ ██████╗  ██████╗██╗  ██╗    ██╗    ██╗██╗███████╗ █████╗ ██████╗ ██████╗ 
██╔══██╗██║    ██╔════╝╚══██╔══╝██╔═══██╗██╔════╝██║ ██╔╝    ██║    ██║██║╚══███╔╝██╔══██╗██╔══██╗██╔══██╗
███████║██║    ███████╗   ██║   ██║   ██║██║     █████╔╝     ██║ █╗ ██║██║  ███╔╝  ███████║██████╔╝██║  ██║
██╔══██║██║    ╚════██║   ██║   ██║   ██║██║     ██╔═██╗     ██║███╗██║██║ ███╔╝  ██╔══██║██╔══██╗██║  ██║
██║  ██║██║    ███████║   ██║   ╚██████╔╝╚██████╗██║  ██╗    ╚███╔███╔╝██║███████╗██║  ██║██║  ██║██████╔╝
╚═╝  ╚═╝╚═╝    ╚══════╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═╝     ╚══╝╚══╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
""")

# Ask the user to input a ticker symbol
ticker = input("Enter a ticker symbol: ".upper())
print("")

# Get today's date as a string in the format YYYY-MM-DD
date_today = datetime.today().strftime('%Y-%m-%d')

# Get the start date from the user, with a default of "2020-01-01"
print("Enter the start date for the stock history download (YYYY-MM-DD),")
date_start = input("      or press enter to use the default of 2020-01-01: ")
if date_start == "":
    date_start = "2020-01-01"
print("")

# Get the end date from the user, with a default of today's date
print("Enter the end date for the stock history download (YYYY-MM-DD)")
date_end = input("      or press enter to use today's date:")
if date_end == "":
    date_end = date_today

print("")
epoch = int(input("Enter the number of epochs you would like the model to train for: "))
print("")

# Add this line to ask if the user wants to see the training results graph
view_graph = input("Would you like to view the training results graph? Type 'y' for yes or press enter for no: ").lower()
print("")


# Check if the ticker is valid and if data is available
valid_ticker = False
while not valid_ticker:
    try:
        # Download historical data for the ticker symbol
        data = yf.download(ticker, start=date_start, end=date_end, progress=False)
        if data.empty:
            raise ValueError("No data available for the provided ticker.")
        valid_ticker = True
    except (KeyError, ValueError) as e:
        print(e)
        ticker = input("Please enter a valid ticker symbol: ")

# Clear terminal screen
# print("\033c", end="")

print("Learning from Previous Stock Data...")

# Use only 'Close' column from the data
data = data[['Close']]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
data.loc[:, 'Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


# Function to split the data into training and testing sets
def split_data(stock, lookback):
    data_raw = stock.to_numpy()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]))  # 20% of data for testing
    train_set_size = data.shape[0] - test_set_size

    # split data into training and testing sets
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


# Define lookback period and split the data
lookback = 20
x_train, y_train, x_test, y_test = split_data(data, lookback)

# Convert dataset to tensors
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

# Define model parameters
input_dim = 1
hidden_dim = 64
num_layers = 2
output_dim = 1
num_epochs = epoch

# Check if CUDA is available and set device to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Define output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Initialize the LSTM model
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
model.to(device)

# Define loss function (Mean Squared Error Loss)
criterion = torch.nn.MSELoss(reduction='mean')
# Define optimizer (Adam optimizer)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
hist = np.zeros(num_epochs)
# Initialize tqdm progress bar
with tqdm(total=num_epochs, unit='epoch', ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
    for t in range(num_epochs):
        # Forward pass
        y_train_pred = model(x_train.to(device))
        # Compute loss
        loss = criterion(y_train_pred, y_train.to(device))
        hist[t] = loss.item()
        # Zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()

        # Update the tqdm progress bar
        pbar.set_postfix(loss=loss.item())
        pbar.update(1)

# Make predictions on test data
y_test_pred = model(x_test.to(device))

# Invert predictions (we had normalized the data earlier)
y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

# Calculate root mean squared error for training and test data
trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))

# Calculate R-squared score for training and test data
train_r2_score = r2_score(y_train[:, 0], y_train_pred[:, 0])
test_r2_score = r2_score(y_test[:, 0], y_test_pred[:, 0])

# Calculate the confidence score based on R-squared score
confidence_score = max(min(100, 100 * test_r2_score), 0)

# Change the plot block like this:
if view_graph == 'y':
    # Create a plot
    plt.figure(figsize=(14,5))

    # Plot the actual prices for both training and test data
    plt.plot(range(y_train.shape[0] + y_test.shape[0]), np.concatenate([y_train, y_test])[:,0], color="blue", label="Actual Price")

    # Plot the predicted prices for the test data
    plt.plot(range(y_train.shape[0], y_train.shape[0] + y_test_pred.shape[0]), y_test_pred[:,0], color="red", label="Predicted Price")

    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


# Print Today's Closing Price
last_close_price_normalized = data['Close'][-1]
last_close_price = scaler.inverse_transform([[last_close_price_normalized]])[0][0]
print("")
print("******************** RESULTS ********************")
print("")
print("Last closing price: ", last_close_price)

# Predict the next day's closing price
next_day_input = torch.from_numpy(data[-lookback:].values).type(torch.Tensor)
next_day_input = next_day_input.unsqueeze(0).to(device)  # add extra batch dimension
next_day_price = model(next_day_input)
next_day_price = scaler.inverse_transform(next_day_price.detach().cpu().numpy())
print("The predicted closing price tomorrow is: ", next_day_price[0][0])

# Calculate the percent difference between the actual and predicted prices.
percent_difference = ((next_day_price[0][0] - last_close_price) / next_day_price[0][0]) * 100

# Print the percent difference
print(f"The percent difference between the actual and predicted prices is: {percent_difference:.2f}%")
print("")
print("*************************************************")
print("")

print("************** AI Suggested Action **************")
print("")
# Print the suggestion based on the percent difference
if percent_difference >= 1:
    print(colors.GREEN + "The model suggests buying." + colors.RESET)
elif percent_difference <= -3:
    print(colors.RED + "The model suggests selling." + colors.RESET)
else:
    print(colors.YELLOW + "The model suggests holding. No action required." + colors.RESET)

# Print the confidence score
print(f"Confidence Score: {confidence_score:.2f}%")
print("")
print("*************************************************")
