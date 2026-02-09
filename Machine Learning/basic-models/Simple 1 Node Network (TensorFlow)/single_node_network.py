import tensorflow as tf
# Define the model
class SimpleNeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense(inputs)
        return x


# Create the model
model = SimpleNeuralNetwork()

# Compile the model
model.compile(optimizer='sgd', loss='mse')

# Provide some example input and output pairs
inputs = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)

# Calculate outputs based on the true relationship
outputs = 0.5 * inputs[:, 0] + 0.3 * inputs[:, 1] + 0.1
outputs = tf.reshape(outputs, (-1, 1))


# Train the model
model.fit(inputs, outputs, epochs=300)

# Make predictions with the model
new_input = tf.constant([[1.0, 2.0]], dtype=tf.float32)
prediction = model.predict(new_input)
print("Prediction:", prediction)

# ... (previous code for training and evaluating the model)

# Save the model
model.save("single_node_network")
