import cv2
from flask import Flask, render_template, request ,jsonify
import tensorflow as tf
import base64, uuid
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['SECRET_KEY'] = "hiboomer"


def elaborate(imagefile):
    session = uuid.uuid4()
    path = f"./static/{session}.jpg"
    with open(path, "wb") as f:
        f.write(imagefile)

    image_object = cv2.imread(path)
    # convert in grayscale
    grey = cv2.cvtColor(image_object.copy(), cv2.COLOR_BGR2GRAY)
    # converts the pixels with a value < 75 in zeroes
    # else they decome 255
    a, thresh = cv2.threshold(grey, 75, 255, cv2.THRESH_BINARY_INV)

    # Resizing that digit to (18, 18)
    resized_digit = cv2.resize(thresh, (18, 18))

    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

    # Adding the preprocessed digit to the list of preprocessed digits
    # preprocessed_digits.append(padded_digit)

    # inp = np.array(preprocessed_digits)
    model = tf.keras.models.load_model("mnist.model")
    # for digit in preprocessed_digits:
    digit = model.predict([padded_digit.reshape(1, 28, 28, 1)])
    image_object = None
    # os.remove(path)
    return np.argmax(digit)

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
                digit = elaborate(encoded_file)
                print(digit)
                return jsonify({"status_code": 200, "digit_guessed": str(digit)})
            else:
                return jsonify({"status_code": 400, "error": 'File not supported'})
        elif imagefile:
            if imagefile.filename[str(imagefile.filename).rfind("."):] in app.config["UPLOAD_EXTENSIONS"]:
                digit = elaborate(imagefile)
                return jsonify({"status_code": 200, "digit_guessed": str(digit)})
        else:
            raise ValueError
    except:
        return jsonify({"status_code": 400, "error": 'File not supported'})

if __name__ == "__main__":
    app.run()
