import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model("single_node_network")

# Example new input data
new_input = tf.constant([[8.0, 9.0]], dtype=tf.float32)

# Make predictions with the loaded model
prediction = loaded_model.predict(new_input)
print("Prediction:", prediction)
