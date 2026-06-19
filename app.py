from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone, timedelta
from openai import OpenAI
import os
model = "openrouter/auto"
api_key = os.getenv("OPENROUTER_API_KEY", "your-api-key")
server_url = "https://ai.hackclub.com/proxy/v1"
GOAL_PROMPT = """You are a fitness coach. You will create a goal based on the user statistics and user activity. It will be completed by the user in a week. Make sure the goal is achievable and realistic for the user and you should include only the goal and nothing else."""


client = OpenAI(api_key=api_key, base_url=server_url)

# 1. Initialize Firebase Admin using your downloaded JSON file
cred = credentials.Certificate('fitness-ysws-firebase-adminsdk-fbsvc-c6dad255a7.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
app = Flask(__name__)

@app.route('/api/fitness-webhook', methods=['POST'])
def fitness_webhook():
    try:
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

        # 2. Calculate Base FitCoins based on sport
        base_fitcoins = 0
        if activity_type == "steps":
            base_fitcoins = value / 100
        elif activity_type == "swimming":
            base_fitcoins = value * 10
        elif activity_type == "running":
            base_fitcoins = value * 20
        elif activity_type == "weight_lifting":
            base_fitcoins = value * 30
            
        # If no coins earned, don't process further
        if base_fitcoins <= 0:
            return jsonify({"message": "No FitCoins earned"}), 200

        # 3. Read Database for Cheating & Streaks
        user_ref = db.collection('Users').document(user_id)
        user_doc = user_ref.get()
        
        current_streak = 0
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            last_logged_at = user_data.get('last_logged_at')
            current_streak = user_data.get('current_streak', 0)
            
            # Anti-Cheat: Check if last log was less than 1 hour ago
            if last_logged_at:
                # Firestore returns datetime objects with timezone info
                time_since_last_log = datetime.now(timezone.utc) - last_logged_at
                if time_since_last_log < timedelta(hours=1):
                    return jsonify({"error": "Cooldown active. Try again later!"}), 429
                    
        # 5. Calculate Final FitCoins with Streak Multiplier
        # Equation: base_points + base_points * (streak * 0.01)
        multiplier_bonus = base_fitcoins * (current_streak * 0.01)
        final_fitcoins = round(base_fitcoins + multiplier_bonus, 2)

        # 6. Update the User's Wallet in Firestore
        if not user_doc.exists:
            # Create a new user profile if this is their first time
            user_ref.set({
                'fitcoins_balance': final_fitcoins,
                'current_streak': 1, # Start their streak!
                'last_logged_at': firestore.SERVER_TIMESTAMP
            })
        else:
            # Update existing user profile safely using Increment
            user_ref.update({
                'fitcoins_balance': firestore.Increment(final_fitcoins),
                'last_logged_at': firestore.SERVER_TIMESTAMP
                # Note: You'll want a cron job to update streaks daily, but we leave it as is for now
            })

        # 6. Save the raw workout data to a subcollection for history
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

if __name__ == '__main__':
    app.run(port=3000, debug=True)
