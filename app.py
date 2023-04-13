import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request

app = Flask(__name__)

model = tf.keras.models.load_model('weights.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'})
    
    file = request.files['file']
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'file type not allowed'})
    
    img = image.load_img(file, target_size=(125, 125))

    x = image.img_to_array(img)

    x = x / 255.0

    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)

    if prediction < 0.5:
        result = 'damaged'
    else:
        result = 'not damaged'
    
    return jsonify({'result': result})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run()
