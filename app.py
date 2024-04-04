import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
from skimage import feature

# Load the model
with open('all.pkl', 'rb') as file:
    model = pickle.load(file)

def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.flatten()

def compute_texture_gradients(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute texture gradients using Sobel operator
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude of gradients
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Compute mean and standard deviation of gradient magnitude
    mean_gradient = np.mean(gradient_magnitude)
    std_gradient = np.std(gradient_magnitude)
    
    # Compute texture gradients feature vector
    texture_gradients = np.array([mean_gradient, std_gradient])
    
    return texture_gradients

def extract_features(image):
    color_hist = extract_color_histogram(image)
    ta = compute_texture_gradients(image)
    edges = extract_edges(image)
    feature_vector = np.concatenate([color_hist, ta, edges])
    return feature_vector

# Create Flask app
app1 = Flask(__name__)

@app1.route("/")
def home():
    return render_template("index.html")

@app1.route("/predict", methods=["POST"])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template("index.html", prediction_text="No file uploaded")

    # Get the uploaded file
    uploaded_file = request.files['file']

    # Check if the file is empty
    if uploaded_file.filename == '':
        return render_template("index.html", prediction_text="No file selected")

    # Check if the file is allowed
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if uploaded_file.filename.split('.')[-1].lower() not in allowed_extensions:
        return render_template("index.html", prediction_text="Invalid file format. Please upload a .png or .jpg file.")

    #re = cv2.resize(image, (224, 224))

    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

    # Print image shape for debugging
    print("Image shape:", image.shape)

    # Preprocess the image
    re = cv2.resize(image, (224, 224))
    preprocessed_image = extract_features(re)

    # Print preprocessed feature vector for debugging
    print("Preprocessed feature vector shape:", preprocessed_image.shape)

    # Make prediction
    prediction = model.predict([preprocessed_image])[0]

    # Map prediction back to label
    number_to_label = {0: "fresh banana", 1: "fresh green apple", 2: "rotten apple", 3: "rotten banana", 4: "fresh red apple",
                        5: "fresh bitter gourd", 6: "fresh capsicum", 7: "fresh orange", 8: "fresh tomato",
                        9: "rotten bitter gourd", 10: "rotten capsicum", 11: "rotten orange"}

    predicted_label = number_to_label[prediction]

    return render_template("index.html", prediction_text="The fruit species is {}".format(predicted_label))

if __name__ == "__main__":
    app1.run(debug=True)
