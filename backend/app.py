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
import google.generativeai as genai

# Gemini API setup
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))  # Make sure to set this environment variable
model = genai.GenerativeModel('gemini-pro')

def generate_skincare_routine(skin_type, skin_issues, age, gender):
    try:
        # Prepare the prompt for the LLM
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

        # Call Gemini API
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating skincare routine: {str(e)}")
        return "Unable to generate skincare routine at this time."

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Get form data
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

        # Final response with recommendations and skincare routine
        return render_template("result.html",
                             image_url=f'/output/{image_id}/output.jpg',
                             skin_condition=skin_condition,
                             skin_issues=skin_issues,
                             recommendations=recommended_products,
                             skincare_routine=skincare_routine,
                             age=age,
                             gender=gender)