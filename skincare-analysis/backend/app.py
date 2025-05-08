from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import os
import uuid
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from dotenv import load_dotenv
import google.generativeai as genai
import markdown

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
        prompt = f"""
Generate a simple and effective skincare routine based on the following details:

- Age: {age}
- Gender: {gender}
- Skin Type: {skin_type}
- Skin Concerns: {', '.join(issues_list)}

•Morning Routine
• [suggest basic steps]

•Evening Routine
• [suggest basic steps]

•Weekly Care
• [mention 1-2 treatments]

•Lifestyle Tips
• [simple advice related to age and gender]

Keep the tone friendly and the suggestions short and easy to follow.

        Format the response in a clear, structured way with bullet points."""

        response = gemini_model.generate_content(prompt)
        routine_markdown = response.text
        routine_html = markdown.markdown(routine_markdown)
        return routine_html
    except Exception as e:
        print(f"Error generating skincare routine: {str(e)}")
        return "Unable to generate skincare routine at this time."

# Load and preprocess the product dataset
product_df = pd.read_csv("product-data.csv")
product_df.fillna('', inplace=True)

#-----------
print("Unique Genders:", product_df['Gender'].unique())
print("Unique Ages:", product_df['Age'].unique())
#-------------

# Clean columns
product_df['Skin Type'] = product_df['Skin Type'].str.lower().str.strip()
product_df['Skin Concerns'] = product_df['Skin Concerns'].apply(lambda x: [i.strip().lower() for i in str(x).split(',')])
product_df['Gender'] = product_df['Gender'].str.lower().str.strip()
product_df['Age'] = product_df['Age'].str.strip()

mlb = MultiLabelBinarizer()
concern_vectors = mlb.fit_transform(product_df['Skin Concerns'])

skin_type_vector = pd.get_dummies(product_df['Skin Type'])
combined_vectors = pd.concat([pd.DataFrame(skin_type_vector), pd.DataFrame(concern_vectors)], axis=1).values

# Upload/output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Roboflow setup
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="gmIokv6xUEW8cJPv4C1Q"
)
ROBOFLOW_MODEL_ID = "my-first-project-y0u0j/3"
DETECTION_LABELS = ['Acne', 'Dark Circle', 'Dark spot', 'Eyebag', 'Mole', 'Redness', 'Wrinkles', 'freckles', 'whiteheads']

# Skin type classification
NUM_CLASSES = 3
SKIN_TYPE_LABELS = ['dry', 'oily', 'normal']
resnet = models.resnet50(pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
resnet.load_state_dict(torch.load('resnet50_best_three_class.pth', map_location='cpu'))
resnet.eval()

@app.route('/')
def index():
    return render_template('index.html')

def classify_skin(image_path):
    try:
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
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
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for prediction in result.get('predictions', []):
            class_id = prediction.get('class_id', -1)
            if 0 <= class_id < len(DETECTION_LABELS):
                label = DETECTION_LABELS[class_id - 1]
                confidence = round(prediction.get('confidence', 0) * 100, 2)
                x = prediction.get('x', 0)
                y = prediction.get('y', 0)
                width = prediction.get('width', 0)
                height = prediction.get('height', 0)

                left = int(x - width / 2)
                top = int(y - height / 2)
                right = int(x + width / 2)
                bottom = int(y + height / 2)

                draw.rectangle([left, top, right, bottom], outline="red", width=2)
                draw.text((left, top - 10), f"{label} ({confidence}%)", fill="red", font=font)

                detections.append({
                    'label': DETECTION_LABELS[class_id - 1],
                    'confidence': confidence,
                    'position': {
                        'x': round(x), 'y': round(y),
                        'width': round(width), 'height': round(height)
                    }
                })

        image.save(image_path)
        return detections
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return []

def recommend_products(skin_type, issues, user_age, user_gender):
    try:
        # Vectorize user profile
        skin_vector = [0] * len(skin_type_vector.columns)
        if skin_type in skin_type_vector.columns:
            skin_vector[skin_type_vector.columns.get_loc(skin_type)] = 1

        issue_vector = [0] * len(mlb.classes_)
        for issue in issues:
            label = issue['label'].lower()
            if label in mlb.classes_:
                issue_vector[list(mlb.classes_).index(label)] = 1

        user_vector = skin_vector + issue_vector

        # Compute cosine similarity
        similarities = cosine_similarity([user_vector], combined_vectors)[0]
        product_df['score'] = similarities

        # Filter top 50 by similarity
        top_products = product_df.sort_values(by='score', ascending=False).head(50)

        # Filter by exact match of gender and age
        filtered_products = top_products[
    ((top_products['Gender'].str.lower() == user_gender) | (top_products['Gender'].str.lower() == 'unisex')) &
    (top_products['Age'].str.strip() == user_age)
]


        return filtered_products[['Label', 'Product Name', 'Brand', 'Price', 'Product Link', 'Image URL']].to_dict(orient='records')
        print("Filtered Products Found:", len(filtered_products))

    except Exception as e:
        print(f"Recommendation error: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import base64
    from io import BytesIO

    try:
        # Get inputs
        image = request.files.get('image')
        captured_data = request.form.get('captured_image')  # base64 from webcam

        if not image and not captured_data:
            return jsonify({'error': 'No image uploaded or captured'}), 400

        # Save image
        image_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.jpg")

        if image and image.filename != '':
            image.save(upload_path)
        elif captured_data:
            try:
                header, encoded = captured_data.split(',', 1)
                binary_data = base64.b64decode(encoded)
                img = Image.open(BytesIO(binary_data))
                img.save(upload_path)
            except Exception as e:
                print(f"Error decoding captured image: {e}")
                return jsonify({'error': 'Failed to process captured image'}), 500
        else:
            return jsonify({'error': 'No valid image input'}), 400

        # Get form inputs
        user_gender = request.form.get("gender", "").strip().lower()
        user_age = request.form.get("age", "").strip()

        # Run predictions
        skin_condition = classify_skin(upload_path)
        skin_issues = detect_skin_issues(upload_path)

        # Generate skincare routine
        skincare_routine = generate_skincare_routine(skin_condition, skin_issues, user_age, user_gender)

        # Save output image with bounding boxes
        output_dir = os.path.join(OUTPUT_FOLDER, image_id)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output.jpg")
        Image.open(upload_path).save(output_path)

        # Log for debugging
        print("User Selected Gender:", user_gender)
        print("User Selected Age:", user_age)

        # Recommend products
        recommended_products = recommend_products(skin_condition, skin_issues, user_age, user_gender)

        return render_template("result.html",
                               image_url=f'/output/{image_id}/output.jpg',
                               skin_condition=skin_condition,
                               skin_issues=skin_issues,
                               recommendations=recommended_products,
                               skincare_routine=skincare_routine)

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
