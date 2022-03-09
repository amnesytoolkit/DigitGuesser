import os
import random

import cv2
from flask import Flask, render_template, request ,jsonify
import tensorflow as tf
import base64, uuid
import numpy as np
import pickle

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['SECRET_KEY'] = "hiboomer"

labels = ['7', '3', 'k', 'S', 'v', 'T', 'm', 'Q', '9', 'a', 'f', 'U', '1', 'J', 'c', '0', 'j', 's', 'l', 'd',
          'I', 'b', 'w', 'r', 'W', 'X', 'C', 'p', '2', '5', 'x', 'n', 'e', '4', 'E', 'V', 'D', 't', 'h', 'Y',
          'M', 'Z', '6', 'G', '8', 'K', 'q', 'o', 'g', 'A', 'H', 'O', 'R', 'u', 'Img', 'i', 'P', 'y', 'z',
          'F', 'N', 'B', 'L']
def elaborate(imagefile):
    session = uuid.uuid4()
    path = f"./static/images/{session}.png"
    with open(path, "wb") as f:
        f.write(imagefile)

    image_object = cv2.imread(path)
    # convert in grayscale
    grey = cv2.cvtColor(image_object.copy(), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, image_object.copy())
    # converts the pixels with a value < 60 in zeroes
    # else they decome 255
    a, thresh = cv2.threshold(grey, 70, 1, cv2.THRESH_BINARY_INV)
    # a, thresh = cv2.threshold(grey, 70, 1, cv2.THRESH_BINARY_INV)

    # Resizing that digit to (18, 18)
    resized_digit = cv2.resize(thresh, (28, 28))

    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    # padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

    # Adding the preprocessed digit to the list of preprocessed digits
    # preprocessed_digits.append(padded_digit)

    # inp = np.array(preprocessed_digits)
    model = tf.keras.models.load_model("mnist.model")
    # for digit in preprocessed_digits:
    data = resized_digit.reshape(1, 28, 28, 1)
    prob = model.predict([data])
    prob = prob[0][np.argmax(prob)]
    digit = labels[np.argmax(model.predict([data]))]
    # save the data on the disk
    # os.remove(path)
    return digit, prob

@app.route("/")
def digitsguesser():
    return render_template("form.html")

@app.route("/api/digits", methods=["POST"])
def endpoint():
    try:
        imagefile = request.files.get('txt_bin_file', False)
        # file sent with a post request
        encoded_file = request.form.get('data_image', False)
        encoded_file_enctype = request.form.get('encoding', False)

        if encoded_file:
            if encoded_file_enctype == "base64":
                # we strip the first 23 chars that in a base64 encoded image
                # are like this:
                # es -> data:image/jpeg;base64,
                encoded_file = base64.b64decode(encoded_file[23:])
                # elabora l'immagine
                digit, prob = elaborate(encoded_file)
                print(digit, " ", prob)
                return jsonify({"status_code": 200, "digit_guessed": str(digit), "probability": str(prob)})
            else:
                return jsonify({"status_code": 400, "error": 'File not supported'})
        elif imagefile:
            if imagefile.filename[str(imagefile.filename).rfind("."):] in app.config["UPLOAD_EXTENSIONS"]:
                digit, prob = elaborate(imagefile)
                print(digit, " ", prob)
                return jsonify({"status_code": 200, "digit_guessed": str(digit), "probability": str(prob)})
        else:
            raise ValueError
    except Exception as e:
        return jsonify({"status_code": 400, "error": 'File not supported'})

if __name__ == "__main__":
    app.run()
