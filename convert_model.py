import tensorflow as tf

model = tf.keras.models.load_model("outputs_dl/model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("outputs_dl/model.tflite", "wb") as f:
    f.write(tflite_model)

print("Done! model.tflite saved.")