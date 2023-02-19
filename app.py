from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

PATIENT_STATE = ['good', 'caution', 'danger']


@app.route("/", methods=['POST'])
def root():
    data = request.get_json()
    row = np.array(data.get('row'), dtype=np.float32)
    print(row)
    interpreter.set_tensor(input_index, [row])
    interpreter.invoke()
    res = interpreter.get_tensor(output_index)
    return jsonify({'patientState': PATIENT_STATE[np.argmax(res)]})


if __name__ == "__main__":
    app.run(debug=True)
