import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model('app/ml/chatbot_model.h5')

# Convert it to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the new small model
with open('app/ml/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted! Your file is now much smaller.")