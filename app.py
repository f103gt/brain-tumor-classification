import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.models as models

app = Flask(__name__)

CONFIG = {
    "IMAGE_SIZE": 224,
    "NUM_CLASSES": 4,
    "MODEL_PATH": "brain_tumour_classification_pytorch/brain_tumor_model.pth",
    "CLASSES": ["glioma", "meningioma", "notumor", "pituitary"]
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(CONFIG["MODEL_PATH"], map_location=device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    image = Image.open(image)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Ensure 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Save the file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)
    
    try:
        predicted_class = predict_image(file_path)
        predicted_label = CONFIG["CLASSES"][predicted_class]
        return jsonify({'predicted_class': predicted_label}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
