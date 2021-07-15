from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys

model = load_model('mnist_app.h5')

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def model_predict(path, model):
    img = cv2.imread(path)

    # shape of the image
    # print(img.shape)

    # let's convert it to gray image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # let's resize
    imgResize = cv2.resize(imgGray, (28, 28))

    # Let's reshape
    final_image = np.expand_dims(imgResize, axis=-1)
    final_image = np.expand_dims(final_image, axis=0)

    # prediction
    predict = model.predict(final_image)

    return predict.argmax()


@app.route('/')
def home():
    return render_template('home.html')


# @app.route('/display/<filename>')
# def display(filename):
#     return redirect(url_for('static', filename))


@app.route('/predict', methods=['POST'])
def predict():

    target = os.path.join(APP_ROOT, 'static/')

    # print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        # print(file)
        file_name = file.filename
        destination = '/'.join([target, file_name])
        # print(destination)
        file.save(destination)

    pred = model_predict(destination, model)

    return render_template('prediction.html', result=pred, filename=file_name)


if __name__ == '__main__':
    app.run()
