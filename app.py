import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "shit-secret-key"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = tf.keras.models.load_model("weights.h5")

@app.route('/predict', methods=['POST'])
def predict():
    
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    files = request.files.getlist('files[]')
     
    errors = {}
    results = []
     
    for file in files:      
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(125, 125))
            x = image.img_to_array(img)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)
            prediction = model.predict(x)
            if prediction < 0.5:
                result = 'damaged'
            else:
                result = 'not damaged'
            results.append(result)
        else:
            errors[file.filename] = 'File type is not allowed'
            results.append(errors)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8880, debug=True)
