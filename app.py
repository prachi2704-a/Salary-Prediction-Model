import joblib
import pandas as pd
import os
import logging
import webbrowser
import threading
from flask import Flask, request, jsonify, render_template

# --- Application Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__, template_folder='templates')


# --- Model Loading ---
MODEL_FILE = 'salary_prediction_model_enhanced.pkl'
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)
try:
    model = joblib.load(model_path)
    logging.info(f"Model '{MODEL_FILE}' loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Could not load model. Error: {e}")
    model = None

# --- Web Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not available.'}), 500
        
    try:
        data = request.get_json(force=True)
        
        # --- ENHANCED VALIDATION LOGIC ---
        experience = int(data['experience'])
        age = int(data['age'])
        
        # Assuming professional experience starts at age 18
        max_possible_experience = age - 18

        # Check if the entered experience is realistic for the given age.
        if experience > max_possible_experience:
            logging.warning(f"Invalid input: Experience ({experience}) is unrealistic for Age ({age}). Max possible is {max_possible_experience}.")
            # Send a specific, helpful error message back to the frontend
            return jsonify({'error': f'For an age of {age}, experience cannot be more than {max_possible_experience} years.'}), 400
        # --- END OF ENHANCED VALIDATION ---

        employee_profile = {
            'Job Title': data['jobTitle'],
            'Years of Experience': experience,
            'Education Level': data['education'],
            'Age': age
        }
        profile_df = pd.DataFrame([employee_profile])
        
        prediction = model.predict(profile_df)
        return jsonify({'predicted_salary': prediction[0]})
        
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid or missing input data. Details: {e}'}), 400
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'Could not process the prediction.'}), 500

if __name__ == '__main__':
    url = "http://127.0.0.1:5001"
    threading.Timer(1.25, lambda: webbrowser.open_new_tab(url)).start()
    app.run(host='0.0.0.0', port=5001, debug=True)