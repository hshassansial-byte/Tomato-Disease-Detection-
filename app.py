from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)


MODEL_PATH = 'TomatoCMDD_Updated.h5'

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"model '{MODEL_PATH}' loaded successfully!")
else:
    print(f"Error: {MODEL_PATH} not found in the project folder!")

# 2. Define Class Names & Solutions
class_names = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
               'Septoria_leaf_spot', 'Target_Spot', 'Two-spotted_spider_mite', 
               'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy']

solutions = {
    'Bacterial_spot': "Spray with copper fungicide and stop watering from above.",
    'Early_blight': "Cut off the bottom yellow leaves and use a fungicide spray.",
    'Late_blight': "Pull out and throw away the plant immediately to save the rest.",
    'Leaf_Mold': "Open up the garden space for better airflow and lower humidity.",
    'Septoria_leaf_spot': "Pick off infected leaves and keep the soil covered with mulch.",
    'Target_Spot': "Use a protective spray and keep the leaves as dry as possible.",
    'Two-spotted_spider_mite': "Blast them off with a strong water hose or use neem oil.",
    'Tomato_Yellow_Leaf_Curl_Virus': "Control Whiteflies with sticky traps.",
    'Tomato_mosaic_virus': "No cure. Remove plant and sanitize tools.",
    'healthy': "Plant is healthy! Maintain regular care."
}

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))
    
    processed_img = prepare_image(image, target_size=(224, 224))
    
    # Prediction
    preds = model.predict(processed_img)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0]) * 100)
    result = class_names[idx]
    
    # low Confidence
    if conf < 60:
        advice = f"Prediction: {result.replace('_', ' ')} (Low Confidence: {conf:.2f}%). Please take a clearer photo."
    else:
        advice = solutions.get(result, "No specific solution found.")

    return jsonify({
        'prediction': result.replace('_', ' '),
        'confidence': f"{conf:.2f}%",
        'solution': advice
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)