from flask import Flask, request, jsonify, render_template, send_from_directory
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone, timedelta
from openai import OpenAI
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# setup the openai client for the fitness coach.
model = "openrouter:free"
api_key = os.getenv("OPENROUTER_API_KEY", "your-api-key")
server_url = "https://ai.hackclub.com/proxy/v1"
GOAL_PROMPT = """You are a fitness coach. You will create a goal based on the user statistics and user activity. It will be completed by the user in a week. Make sure the goal is achievable and realistic for the user and you should include only the goal and nothing else."""

client = OpenAI(api_key=api_key, base_url=server_url)

# Firebase initialization moved below TensorFlow to prevent gRPC deadlock

# ==============================================================================
# AI MODEL SETUP
# Load the brain ONE TIME when the server starts so we don't crash the RAM
# ==============================================================================
print("Loading AI Brain into memory...")
MODEL = tf.keras.models.load_model('ml_pipeline/exercise_model.keras')

with open('ml_pipeline/scaler.pkl', 'rb') as f:
    SCALER = pickle.load(f)

LABEL_CLASSES = np.load('ml_pipeline/label_classes.npy', allow_pickle=True)
print("AI Brain Loaded Successfully! Ready for webcam data.")

# 1. Initialize Firebase Admin using your downloaded JSON file
cred = credentials.Certificate('fitness-ysws-firebase-adminsdk-fbsvc-c6dad255a7.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)


@app.route('/workout')
def workout_page():
    # Welcome to the future! This serves our shiny new AI Anti-Cheat workout page.
    return render_template('workout.html')


@app.route('/api/predict', methods=['POST'])
def predict_pose():
    # this gets hit 30 times a second by the webcam, so it needs to be FAST
    try:
        landmarks = request.json.get('landmarks')
        if not landmarks or len(landmarks) < 33:
            return jsonify({"error": "No human detected"}), 400
            
        # Extract the exact 132 features (x, y, z, visibility for 33 landmarks)
        row = []
        for lm in landmarks:
            row.extend([lm['x'], lm['y'], lm['z'], lm.get('visibility', 0.0)])
            
        # Reshape to a 2D array: 1 row, 132 columns
        # wrap in a dataframe so sklearn doesnt yell about missing feature names
        X_raw = pd.DataFrame([row], columns=SCALER.feature_names_in_)
        
        # Scale the webcam data EXACTLY how the training data was scaled
        X_scaled = SCALER.transform(X_raw)
        
        # Ask the AI what exercise this is!
        predictions = MODEL.predict(X_scaled, verbose=0)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        exercise_name = LABEL_CLASSES[class_index]
        
        return jsonify({
            "exercise": exercise_name,
            "confidence": confidence
        }), 200
        
    except Exception as e:
        print(f"Prediction crashed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/fitness-webhook', methods=['POST'])
def fitness_webhook():
    try:
        # 0. Anti-Cheat: API Secret Verification
        api_key_header = request.headers.get("X-API-KEY")
        if api_key_header != "hackclub-fitness":
            return jsonify({"error": "Unauthorized. Nice try, hacker!"}), 401
            
        webhook_data = request.json
        
        # Check if the iPhone sent empty data or forgot the application/json header
        if not webhook_data:
            print("Error: Received empty or non-JSON data. Make sure Content-Type is application/json!")
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        # 1. Extract JSON Variables
        user_id = webhook_data.get('user_id')
        activity_type = webhook_data.get('activity_type')
        
        # 2. Extract the actual numbers
        raw_value = webhook_data.get('value', 0)
        value = 0
        try:
            # Fun fact: Apple Shortcuts sends steps in tiny chunks throughout the day
            # If it's a giant list of numbers separated by newlines, we sum them all up!
            if isinstance(raw_value, str) and '\n' in raw_value:
                chunks = raw_value.split('\n')
                for chunk in chunks:
                    if chunk.strip():
                        value += float(chunk.strip())
            else:
                value = float(raw_value)
                
        except (ValueError, TypeError):
            print("Warning: Could not convert the health data into a number.")
            
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        # 3. Read Database for Cheating & Streaks
        user_ref = db.collection('Users').document(user_id)
        user_doc = user_ref.get()
        
        current_streak = 0
        previous_steps = 0
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            last_logged_at = user_data.get('last_logged_at')
            current_streak = user_data.get('current_streak', 0)
            
            # Check if they already logged steps today
            if last_logged_at and last_logged_at.date() == datetime.now(timezone.utc).date():
                previous_steps = user_data.get('last_steps_today', 0)

        # 4. DELTA MATH & DAILY CAPS
        # Anti-Cheat: Maximum 30,000 steps per day
        MAX_DAILY_STEPS = 30000
        if value > MAX_DAILY_STEPS:
            value = MAX_DAILY_STEPS # Cap their total steps
            
        new_steps = value - previous_steps
        if new_steps <= 0:
            return jsonify({"message": "No new steps taken since last sync"}), 200
            
        # 5. Calculate Base FitCoins based on sport
        base_fitcoins = 0
        if activity_type == "steps":
            base_fitcoins = new_steps / 100
        elif activity_type == "swimming":
            base_fitcoins = new_steps * 10
        elif activity_type == "running":
            base_fitcoins = new_steps * 20
        elif activity_type == "weight_lifting":
            base_fitcoins = new_steps * 30
            
        # 6. Calculate Final FitCoins with Streak Multiplier
        multiplier_bonus = base_fitcoins * (current_streak * 0.01)
        
        # Anti-Cheat: Eliminate floating point decimals! 
        final_fitcoins = int(round(base_fitcoins + multiplier_bonus))

        # 7. Update the User's Wallet in Firestore
        if not user_doc.exists:
            # Create a new user profile if this is their first time
            user_ref.set({
                'fitcoins_balance': final_fitcoins,
                'current_streak': 1, # Start their streak!
                'last_logged_at': firestore.SERVER_TIMESTAMP,
                'last_steps_today': value
            })
        else:
            # Update existing user profile safely using Increment
            user_ref.update({
                'fitcoins_balance': firestore.Increment(final_fitcoins),
                'last_logged_at': firestore.SERVER_TIMESTAMP,
                'last_steps_today': value
            })

        # 8. Save the raw workout data to a subcollection for history
        workout_ref = user_ref.collection('Workouts').document()
        webhook_data['receivedAt'] = firestore.SERVER_TIMESTAMP
        webhook_data['fitcoins_earned'] = final_fitcoins
        workout_ref.set(webhook_data)
        
        print(f"Awarded {final_fitcoins} coins to {user_id}!")
        return jsonify({
            "message": "Webhook processed and saved",
            "fitcoins_awarded": final_fitcoins
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/api/fitness-coach", methods=["POST"])
def coach_route():
    try:
        data = request.json
        user_id = data.get("user_id")
        user_stats = data.get("user_stats", "")
        user_data = data.get("user_data", "")
        
        goal_text = get_path(user_stats, user_data)
        
        # Save the generated goal to the user's database
        if user_id:
            db.collection('Users').document(user_id).collection('Goals').document().set({
                'goal': goal_text,
                'createdAt': firestore.SERVER_TIMESTAMP
            })
            
        return jsonify({"goal": goal_text}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
        

def get_path(user_stats, user_data):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": GOAL_PROMPT},
                {"role": "user", "content": f"User Stats: {user_stats}\nUser Data: {user_data}\nGenerate a path for this user that they can complete in a week."}
            ],
            model=model,
            stream=False
        )
        message = response.choices[0].message.content
        return message
    except Exception as e:
        print(f"AI Error: {e}")
        return "Keep pushing! Try to beat your steps from yesterday."

@app.route('/test_videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('test_videos', filename)

if __name__ == '__main__':
    # run the server!
    app.run(port=3000, debug=True, use_reloader=False, threaded=False)
