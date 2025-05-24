import os
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)

CONFIG = {
    "MODELS": {
        "brain_tumor": {
            "IMAGE_SIZE": 224,
            "NUM_CLASSES": 4,
            "MODEL_PATH": "brain_tumour_classification_pytorch/brain_tumor_model.pth",
            "CLASSES": ["glioma", "meningioma", "notumor", "pituitary"],
            "TITLE": "Brain Tumor Classification"
        },
        "pneumonia": {
            "IMAGE_SIZE": 224,
            "NUM_CLASSES": 2,
            "MODEL_PATH": "pneumonia_classification/pneumonia_resnet50_final.pth",
            "CLASSES": ["NORMAL", "PNEUMONIA"],
            "TITLE": "Pneumonia Detection"
        }
    },
    "UPLOAD_FOLDER": "uploads"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "brain_tumor": torch.load(CONFIG["MODELS"]["brain_tumor"]["MODEL_PATH"], map_location=device),
    "pneumonia": torch.load(CONFIG["MODELS"]["pneumonia"]["MODEL_PATH"], map_location=device)
}

for model in models.values():
    model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image, model_type):
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = models[model_type](image)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

@app.route('/')
def index():
    return render_template('index.html', models=CONFIG["MODELS"])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'brain_tumor')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if model_type not in CONFIG["MODELS"]:
        return jsonify({'error': 'Invalid model type'}), 400

    if not os.path.exists(CONFIG["UPLOAD_FOLDER"]):
        os.makedirs(CONFIG["UPLOAD_FOLDER"])

    filename = secure_filename(file.filename)
    file_path = os.path.join(CONFIG["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    try:
        predicted_class = predict_image(file_path, model_type)
        predicted_label = CONFIG["MODELS"][model_type]["CLASSES"][predicted_class]
        return jsonify({
            'predicted_class': predicted_label,
            'model_type': model_type
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
