from flask import Flask, request, render_template, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import os

# Load the trained model
model = load_model('animal_classifier_model.h5')

# Map class index to class label
class_labels = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly',
    'cat', 'caterpillar', 'cheetah', 'chimpanzee', 'cockroach', 'cow', 'coyote',
    'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck',
    'eagle', 'elephant', 'flamingo', 'fox', 'fly', 'goose','grasshopper','hare','hedgehog', 'goat', 
    'goldfish','gorilla', 'hamster', 'hornbill','hyena','jaguar','jellyfish','ladybug','mosquito','moth',
    'hippopotamus', 'horse', 'hummingbird', 'kangaroo','koala', 'leopard', 'lion', 'lizard', 'lobster', 'okapi',
    'orangutan','otter','ox','oyster','pelecaniformes','pig', 'mouse',
    'octopus', 'owl', 'panda', 'parrot', 'penguin','pigeon','porcupine','possum','raccoon','rat','reindeer',
    'rhinoceros','sandpiper','seahorse','seal','shark','sheep','snake','sparrow','squid','squirrel','starfish',
    'swan','tiger','turkey','turtle','whale','wolf','wombat','woodpecker','zebra']  # Ensure it matches the model's classes

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if not allowed_file(file.filename):
        return "Invalid file type. Only .png, .jpg, .jpeg allowed.", 400

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    # Preprocess the image
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Generate the URL for the uploaded image
    image_url = url_for('static', filename='uploads/' + secure_filename(file.filename))

    return render_template('result.html', predicted_label=predicted_label, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
