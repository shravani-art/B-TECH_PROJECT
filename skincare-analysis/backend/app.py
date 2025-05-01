from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import os
import uuid
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Gemini API setup
try:
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please set the GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"Gemini setup error: {e}")
    gemini_model = None

def generate_skincare_routine(skin_type, skin_issues, age, gender):
    if not gemini_model:
        return "Gemini model not available. Please check configuration."
    try:
        issues_list = [issue['label'] for issue in skin_issues]
        prompt = f"""Based on the following skin analysis:
        Age: {age} years
        Gender: {gender}
        Skin Type: {skin_type}
        Skin Concerns: {', '.join(issues_list)}

        Please provide a detailed skincare routine including:
        1. Morning routine
        2. Evening routine
        3. Weekly treatments
        4. Lifestyle recommendations

        Consider the person's age and gender in your recommendations.
        Format the response in a clear, structured way with bullet points."""

        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating skincare routine: {str(e)}")
        return "Unable to generate skincare routine at this time."

# Load and preprocess the product dataset
product_df = pd.read_csv("final_cleaned.csv")

# Fill missing values
product_df.fillna('', inplace=True)

# Prepare skin type and concern binarization
product_df['Skin Type'] = product_df['Skin Type'].str.lower().str.strip()
product_df['Skin Concerns'] = product_df['Skin Concerns'].apply(lambda x: [i.strip().lower() for i in str(x).split(',')])

mlb = MultiLabelBinarizer()
concern_vectors = mlb.fit_transform(product_df['Skin Concerns'])

# Create combined feature matrix
skin_type_vector = pd.get_dummies(product_df['Skin Type'])
combined_vectors = pd.concat([pd.DataFrame(skin_type_vector), pd.DataFrame(concern_vectors)], axis=1).values

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
NUM_CLASSES = 3
SKIN_TYPE_LABELS = ['dry', 'oily' , 'normal']

# Load ResNet50 model
resnet = models.resnet50(pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
resnet.load_state_dict(torch.load('resnet50_best_three_class.pth', map_location='cpu'))
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

def recommend_products(skin_type, issues):
    try:
        # Build user profile
        skin_vector = [0] * len(skin_type_vector.columns)
        if skin_type in skin_type_vector.columns:
            skin_vector[skin_type_vector.columns.get_loc(skin_type)] = 1

        issue_vector = [0] * len(mlb.classes_)
        for issue in issues:
            issue = issue['label'].lower()
            if issue in mlb.classes_:
                issue_vector[list(mlb.classes_).index(issue)] = 1

        user_vector = skin_vector + issue_vector

        # Compute cosine similarity
        similarities = cosine_similarity([user_vector], combined_vectors)[0]
        product_df['score'] = similarities
        top_products = product_df.sort_values(by='score', ascending=False).head(50)

        return top_products[['Label', 'Product Name', 'Brand', 'Price', 'Product Link','Image URL']].to_dict(orient='records')
    except Exception as e:
        print(f"Recommendation error: {e}")
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
        age = request.form.get('age', type=int)
        gender = request.form.get('gender', '').lower()

        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not age or not gender:
            return jsonify({'error': 'Age and gender are required'}), 400


        # Save uploaded image
        image_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.jpg")
        image.save(upload_path)

        # Get predictions
        skin_condition = classify_skin(upload_path)
        skin_issues = detect_skin_issues(upload_path)
        # Generate skincare routine with age and gender
        skincare_routine = generate_skincare_routine(skin_condition, skin_issues, age, gender)

        # Copy original image to output for display
        output_dir = os.path.join(OUTPUT_FOLDER, image_id)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output.jpg")
        Image.open(upload_path).save(output_path)

        recommended_products = recommend_products(skin_condition, skin_issues)

        # Final response with recommendations
        return render_template("result.html",
                             image_url=f'/output/{image_id}/output.jpg',
                             skin_condition=skin_condition,
                             skin_issues=skin_issues,
                             recommendations=recommended_products,
                             skincare_routine=skincare_routine,
                             age=age,
                             gender=gender)

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
