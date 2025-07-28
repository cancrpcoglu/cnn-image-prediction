from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('voc2012_multilabel_model.keras')

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

def prepare_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return render_template('index.html', error="Dosya yüklenmedi")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="Dosya seçilmedi")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img_array = prepare_image(filepath)
    pred = model.predict(img_array)[0]
    class_index = np.argmax(pred)
    confidence = float(np.max(pred)) * 100
    predicted_class = class_names[class_index]

    return render_template('result.html',
                           prediction=predicted_class,
                           probability=round(confidence, 2),
                           image_path=filepath)


if __name__ == '__main__':
    app.run(debug=True)
