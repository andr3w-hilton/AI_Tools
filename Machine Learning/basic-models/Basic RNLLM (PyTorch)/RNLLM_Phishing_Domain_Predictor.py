# Import required modules
import torch  # Deep learning library
from torch import nn  # Base class for all neural network modules
from torch.nn.utils.rnn import pad_sequence  # Function to pad a sequence
from torch.nn.functional import cross_entropy  # Compute Cross-Entropy loss
from torch.utils.data import DataLoader  # DataLoader class provides an iterator over a dataset
from torch.optim import Adam  # Adam optimizer
import numpy as np  # Library for mathematical functions
import os  # OS module provides a way of using operating system dependent functionality
import time  # Time access and conversions module
import random  # For generating random numbers
import string  # Contains a number of useful constants and classes, as well as some deprecated legacy functions
import matplotlib.pyplot as plt  # Graph plotting library
import json  # Library to work with JSON data
from matplotlib.ticker import FuncFormatter  # Provides a major formatter using arbitrary functions

CONSTANT_SEPARATOR = "*********************************************************"

# Select appropriate computation device, prefer GPU (cuda) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the text file into a string
# text = open('domains.txt', 'r').read()
text = open('domains.txt', 'r').read().replace('\n', '^') + '^'

lines = text.split('^')

# Print the first 5 entries
for i in range(5):
    print(f"Entry {i+1}: {lines[i]}^")

# Extract unique characters and sort them
vocab = sorted(set(text))



# Hyper Parameters:

# Model Architecture Parameters
vocab_size = len(vocab)  # Size of the vocabulary
embedding_dim = 256  # Dimensionality of the character embeddings
rnn_units = 1024  # Number of "neurons" in the GRU Layer (aka hidden_dim) 1024
num_layers = 1  # Number of layers in the Network

# Data Processing Parameters
seq_length = 100  # The length of the sequence input to the model
BUFFER_SIZE = 10000  # The buffer size for shuffling the dataset

# Training Configuration Parameters
num_workers = 12  # Number of CPU Cores to utilise in the training
learning_rate = 1e-3  # Controls the step size during gradient descent (1e-3 = 0.001)
BATCH_SIZE = 32  # Controls the number of training samples worked through before updating the models internal parameters
EPOCHS = 50  # The number of times the model will go through the entire training dataset

# Data Regularisation Parameters
dropout_rate = 0.5  # Reduces over-fitting by randomly setting a fraction of input units to 0
weight_decay = 1e-5  # Helps prevent the weights from growing too large by adding a penalty to the loss function
clip_value = 1  # Used to prevent exploding gradients, set too low, it could hinder the learning process


print("")
print(CONSTANT_SEPARATOR)
print(f'Identified {len(vocab)} unique characters within the Training Data')
print(CONSTANT_SEPARATOR)
print("")

# Create mapping from unique characters to indices (integers) and vice versa
char2idx = {u:i for i, u in enumerate(vocab)}  # Character to index mapping
idx2char = np.array(vocab)  # Index to character mapping

# Convert the entire text to its corresponding integer indices
text_as_int = np.array([char2idx[c] for c in text])


# Function to convert a list of indices to text
def text_from_ids(ids):
    return ''.join(idx2char[id] for id in ids)


# Calculate number of examples per epoch
examples_per_epoch = len(text) // (seq_length + 1)

# Convert the entire text to sequences of 'seq_length' characters each
char_dataset = list(text_as_int)
sequences = [torch.tensor(char_dataset[i:i+seq_length+1]) for i in range(0, len(char_dataset), seq_length+1)]

# Each sequence is split into input (all characters except last) and target (all characters except first)
dataset = [(seq[:-1], seq[1:]) for seq in sequences]


# Define function for padding sequences in a batch
def collate_fn(batches):
    inputs = [item[0] for item in batches]  # Separate inputs
    targets = [item[1] for item in batches]  # Separate targets
    # Pad sequences with 0s to the max length sequence in a batch
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0).long()

    return inputs_padded, targets_padded


# Define DataLoader for the training set
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=collate_fn)


# Define custom model class, subclass of nn.Module
class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, dropout_rate):
        super(MyModel, self).__init__()  # Call the init function of nn.Module parent class
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Define Embedding layer
        self.gru = nn.GRU(embedding_dim, rnn_units, num_layers=num_layers, batch_first=True)  # Define GRU layer
        self.dropout = nn.Dropout(dropout_rate)  # Add a dropout layer
        self.dense = nn.Linear(rnn_units, vocab_size)  # Define final Dense layer

    def forward(self, x, states=None):  # Forward pass function
        x = self.embedding(x)  # First, pass through Embedding layer
        if states is None:  # If no states are provided, pass only x to GRU
            x, states = self.gru(x)
        else:  # If states are provided, pass both x and states to GRU
            x, states = self.gru(x, states)
        x = self.dense(x)  # Finally, pass through Dense layer

        return x, states


# Instantiate the model and define the optimizer
model = MyModel(vocab_size, embedding_dim, rnn_units, dropout_rate)
# Use Adam as optimizer with a Learning Rate defined in the hyperparams
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Training loop
print(CONSTANT_SEPARATOR)
print("Ingesting The Complete Training Data")
print(CONSTANT_SEPARATOR)
print("")


# Create an empty list to store the loss values after each epoch.
losses = []

# Set the patience level (number of epochs to wait) before early stopping.
patience = 3

# Initialize a counter that will keep track of epochs.
counter = 0

# Initialize a variable to store the best loss value. This will be used to compare the loss
# in the current epoch with the lowest loss seen so far during training.
best_loss = None

# Start a loop that will iterate for a total number epochs
for epoch in range(EPOCHS):

    # Initialize a variable to keep track of the cumulative loss for the current epoch.
    running_loss = 0

    # Start a nested loop that will iterate over each batch in the data loader.
    for batch, (input_example, target_example) in enumerate(dataloader):
        optimizer.zero_grad()  # Reset all computed gradients to zero.
        predictions, _ = model(input_example)  # Pass the input data through the model (forward pass)
        # Compute the loss between the predictions and the actual targets.
        loss = cross_entropy(predictions.view(-1, vocab_size), target_example.view(-1).long())
        # Compute the gradients for each parameter (backward pass).
        loss.backward()
        # Clip the gradient norms to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        # Update the parameters of the model based on the computed gradients.
        optimizer.step()
        # Add the current batch's loss to the running total for this epoch.
        running_loss += loss.item()

    # Calculate the average loss for this epoch
    avg_epoch_loss = running_loss / len(dataloader)
    # Print out the average loss for this epoch.
    print(f"Epoch {epoch+1} / {EPOCHS}, Average Loss: {avg_epoch_loss}")
    # Add the average loss for this epoch to the list of losses.
    losses.append(avg_epoch_loss)

    # Early stopping check.

    # If this is the first epoch (i.e., best_loss is still None), store the average loss as the best loss.
    if best_loss is None:
        best_loss = avg_epoch_loss
    # If the average loss for this epoch is worse (higher) than the best loss seen so far,
    # increment the counter for the number of epochs with no improvement.
    elif avg_epoch_loss > best_loss:
        counter += 1
        # Print a message showing the current count of epochs with no improvement.
        print(f'EarlyStopping Warning: {counter} out of {patience}')

        # If there are epochs with no improvement print that we're stopping early,
        if counter >= patience:
            print("")
            print('Stopping the training early to prevent OverFitting')
            break
    # If the average loss for this epoch is better (lower) than the best loss seen so far,
    # update the best loss to this epoch's average loss, and reset the counter for the number
    # of epochs with no improvement.
    else:
        best_loss = avg_epoch_loss
        counter = 0


# model = torch.load("Domains_LLM.pth") # saves the whole model
# torch.save(model.state_dict(), "model_parameters.pth") # saves just the parameters from the model


# Define function to generate domain names
# def generate_domain(random_character):
#     start_string = random.choice(string.ascii_lowercase)
#
#     # Convert start string to indices
#     input_eval = [char2idx[s] for s in start_string]
#     input_eval = torch.tensor(input_eval).unsqueeze(0).long().to(device)
#
#     domain_generated = []  # Empty list to store generated characters
#
#     model.eval()  # Set model to evaluation mode
#
#     with torch.no_grad():  # No gradient computation
#         hidden = None  # Initialize hidden state to None
#         while True:
#             output, hidden = model(input_eval, hidden)  # Forward pass
#             predicted_id = torch.argmax(output, dim=-1)  # Get predicted character ID
#
#             predicted_id_int = predicted_id.item()  # Get scalar value of predicted ID
#
#             # Use the predicted character as the next input to the model
#             input_eval = torch.tensor([[predicted_id_int]], dtype=torch.long).to(device)
#
#             predicted_char = idx2char[predicted_id_int]  # Convert predicted ID to character
#             if predicted_char == "\n" or predicted_char == " ":
#                 break
#
#             domain_generated.append(predicted_char)  # Append predicted character to list
#
#     return start_string + ''.join(domain_generated)  # Return generated domain
#
#
# print("")
# print(CONSTANT_SEPARATOR)
# print("Predicted Future URL's")
# print(CONSTANT_SEPARATOR)
# print("")
#
# # Generate unique domains
# generated_domains = set()  # Use a set to ensure domains are unique
# while len(generated_domains) < 6:  # How many Domains do you want predicted -1
#     random_character = random.choice(string.ascii_lowercase)  # Choose a random starting character
#     domain = generate_domain(random_character)  # Generate domain
#     domain = domain.strip()  # Remove leading/trailing whitespace
#     if domain and domain not in generated_domains:  # If the domain is unique and not empty
#         generated_domains.add(domain)  # Add domain to set
#         print(domain)  # Print domain

def generate_domain(random_character, max_length=35):
    start_string = random_character

    # Convert start string to indices
    input_eval = [char2idx[s] for s in start_string]
    input_eval = torch.tensor(input_eval).unsqueeze(0).long().to(device)

    domain_generated = []  # Empty list to store generated characters

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # No gradient computation
        hidden = None  # Initialize hidden state to None
        for _ in range(max_length):
            output, hidden = model(input_eval, hidden)  # Forward pass
            output = torch.nn.functional.softmax(output, dim=-1)
            predicted_id = torch.argmax(output, dim=-1)  # Get predicted character ID

            predicted_id_int = predicted_id.item()  # Get scalar value of predicted ID

            # Use the predicted character as the next input to the model
            input_eval = torch.tensor([[predicted_id_int]], dtype=torch.long).to(device)

            predicted_char = idx2char[predicted_id_int]  # Convert predicted ID to character

            # If an end of line character is encountered, stop the generation
            if predicted_char == "^":
                break

            # Append predicted character to list if it's not a newline or space
            if predicted_char != "\n" and predicted_char != " ":
                domain_generated.append(predicted_char)

    # If no ^ token is found, return the generated domain
    return start_string + ''.join(domain_generated)


print("")
print(CONSTANT_SEPARATOR)
print("Predicted Future URL's")
print(CONSTANT_SEPARATOR)
print("")

# Generate unique domains
generated_domains = set()  # Use a set to ensure domains are unique
while len(generated_domains) < 6:  # How many Domains do you want predicted -1
    random_character = random.choice(string.ascii_lowercase)  # Choose a random starting character
    domain = generate_domain(random_character)  # Generate domain
    domain = domain.strip()  # Remove leading/trailing whitespace
    if domain and domain not in generated_domains:  # If the domain is unique and not empty
        generated_domains.add(domain)  # Add domain to set
        print(domain)  # Print domain
