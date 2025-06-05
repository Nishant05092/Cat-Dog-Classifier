import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # uploads inside static

model = load_model('model/model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 256, 256, 3)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = preprocess_image(file_path)
        prediction = model.predict(image)

        label = 'Dog 🐶' if prediction[0][0] > 0.5 else 'Cat 🐱'

        return render_template('result.html', prediction=label, image_name=filename)

    else:
        return redirect('/')

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    app.run(debug=True)

