from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from keras.applications.resnet import preprocess_input

app = Flask(__name__)

model = tf.keras.models.load_model('multi_output_model.h5')
IMAGE_DIMS = (60, 60, 3)

def load_image_inference(imagePath):
    image = cv2.imdecode(np.frombuffer(imagePath, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    return image
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"].read()
    image = load_image_inference(file)
    (categoryProba, genderProba, ageProba, colorProba) = model.predict(np.expand_dims(image, axis=0))
    categoryIdx = categoryProba[0].argmax()
    genderIdx = genderProba[0].argmax()
    ageIdx = ageProba[0].argmax()
    colorIdx = colorProba[0].argmax()
    categoryLabel = articleTypeLB.classes_[categoryIdx]
    genderLabel = genderLB.classes_[genderIdx]
    ageLabel = baseColourLB.classes_[ageIdx]
    colorLabel = seasonLB.classes_[colorIdx]
    response = {
        "category": {
            "label": categoryLabel,
            "probability": categoryProba[0][categoryIdx]
        },
        "gender": {
            "label": genderLabel,
            "probability": genderProba[0][genderIdx]
        },
        "age": {
            "label": ageLabel,
            "probability": ageProba[0][ageIdx]
        },
        "color": {
            "label": colorLabel,
            "probability": colorProba[0][colorIdx]
        }
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True,port=5000)
