import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Create flask instance
app = Flask(__name__)
"""class_labels = ['Cercospora leaf spot (Gray leaf spot)',
                'Common rust',
                'Northern Leaf Blight',
                'healthy']"""

class_labels = ['Healthy', 'Not Healthy']


class_preventive_measures = ["Your crop is healthy" ,  "Please consult the local agriculture officers"]


img_rows, img_cols = 224, 224
image_size = [224, 224, 3]


def get_model():
    global model
    model = load_model('DensenetModel.h5') #DensenetModel.h5
    
    print(" * Model loaded!")


# Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check(path):
    # prediction
    img = load_img(path, target_size=image_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.reshape(-1, 224, 224, 3)
    x = x.astype('float32') / 255
    z = model.predict(x)
    index = np.argmax(z)

    accuracy = int(np.array(z).max() * 100)
    return [index, accuracy]


get_model()


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if (file):
            try:
                if file and allowed_file(file.filename):
                    filename = file.filename
                    file_extension = filename.split('.')[-1]
                    file_path = os.path.join('static/images', "testing-image."+file_extension)
                    file.save(file_path)
                    result = check(file_path)
                    # Predict the class of an image
                    if int(result[1])>= 50 :
                            disease_name = class_labels[0]
                            accuracy = result[1]
                            preventive_measures=class_preventive_measures[0]
                    else:                                                  
                            disease_name = class_labels[1]
                            accuracy = 100-int(result[1])
                            preventive_measures=class_preventive_measures[1]
                            


                        

                    #disease_name = class_labels[result[0]]

                    

                    return render_template('predict.html',
                                           disease_name=disease_name,
                                           user_image=file_path,
                                           accuracy=accuracy,
                                           preventive_measures=preventive_measures)
            except Exception as e:
                return "Error : " + str(e)

        else:
            eMessage = "Please Upload the diseased file"
            return redirect(url_for('predict', error = eMessage))

    elif (request.method == 'GET'):
        # redirect(url_for("home"))
        return redirect(url_for('predict'))

@app.route('/download-image/<path:filename>')
def download(filename):
    return send_from_directory('static', filename, as_attachment=True, mimetype='image/jpg', attachment_filename=(str(filename) + '.jpg'))


if __name__ == "__main__":
    app.run(debug=True)
