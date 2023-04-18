import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import datetime

app = Flask(__name__)
CORS(app, support_credentials=True)
app.secret_key = "shit-secret-key"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = tf.keras.models.load_model("weights.h5")

@app.route('/predict', methods=['POST'])
@cross_origin(origin='*')
def predict():
    if 'files' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    files = request.files.getlist('files')
    manufacturing = int(request.form.get("manufacturing"))
    showroomPrice = int(request.form.get("price"))
    price = showroomPrice

    today = datetime.date.today()
    year = today.year
    
    for i in range(year-manufacturing+1):
        price -= price*0.1
    
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
                price -= price*0.1
        else:
            return jsonify("File type is not allowed"), 400

    if(price < showroomPrice*0.1):
        return jsonify(showroomPrice*0.1), 200

    return jsonify(price), 200

if __name__ == '__main__':
    app.run(debug=True)
