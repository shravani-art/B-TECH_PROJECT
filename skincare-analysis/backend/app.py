from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import os
import uuid
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Define upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Roboflow client setup
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="gmIokv6xUEW8cJPv4C1Q"
)
ROBOFLOW_MODEL_ID = "my-first-project-y0u0j/3"
DETECTION_LABELS = ['Acne', 'Dark Circle', 'Dark spot', 'Eyebag', 'Mole', 'Redness', 'Wrinkles', 'freckles', 'whiteheads']

# Skin type classification setup
NUM_CLASSES = 2
SKIN_TYPE_LABELS = ['dry', 'oily']

# Load ResNet50 model
resnet = models.resnet50(pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
resnet.load_state_dict(torch.load('resnet50_best_skin_classifier.pth', map_location='cpu'))
resnet.eval()

def classify_skin(image_path):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = resnet(input_tensor)
            _, predicted = torch.max(output, 1)
        return SKIN_TYPE_LABELS[predicted.item()]
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return "unknown"

def detect_skin_issues(image_path):
    try:
        result = CLIENT.infer(image_path, model_id=ROBOFLOW_MODEL_ID)
        detections = []
        
        for prediction in result.get('predictions', []):
            class_id = prediction.get('class_id', -1)
            if 0 <= class_id < len(DETECTION_LABELS):
                detections.append({
                    'label': DETECTION_LABELS[class_id],
                    'confidence': round(prediction.get('confidence', 0) * 100, 2),
                    'position': {
                        'x': round(prediction.get('x', 0)),
                        'y': round(prediction.get('y', 0))
                    }
                })
        
        return detections
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded image
        image_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.jpg")
        image.save(upload_path)

        # Get predictions
        skin_condition = classify_skin(upload_path)
        skin_issues = detect_skin_issues(upload_path)

        # Copy original image to output for display
        output_dir = os.path.join(OUTPUT_FOLDER, image_id)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output.jpg")
        Image.open(upload_path).save(output_path)

        return render_template("result.html",
                             image_url=f'/output/{image_id}/output.jpg',
                             skin_condition=skin_condition,
                             skin_issues=skin_issues)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/output/<image_id>/<filename>')
def serve_output(image_id, filename):
    try:
        return send_from_directory(os.path.join(OUTPUT_FOLDER, image_id), filename)
    except FileNotFoundError:
        return "Image not found", 404

if __name__ == '__main__':
    app.run(debug=True)