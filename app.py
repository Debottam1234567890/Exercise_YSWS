from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone, timedelta

# 1. Initialize Firebase Admin using your downloaded JSON file
cred = credentials.Certificate('fitness-ysws-firebase-adminsdk-fbsvc-c6dad255a7.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
app = Flask(__name__)

@app.route('/api/fitness-webhook', methods=['POST'])
def fitness_webhook():
    try:
        webhook_data = request.json
        
        # 1. Extract JSON Variables
        user_id = webhook_data.get('user_id')
        activity_type = webhook_data.get('activity_type')
        value = webhook_data.get('value', 0)
        
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
                    
        # 4. Calculate Final FitCoins with Streak Multiplier
        # Equation: base_points + base_points * (streak * 0.01)
        multiplier_bonus = base_fitcoins * (current_streak * 0.01)
        final_fitcoins = base_fitcoins + multiplier_bonus

        # 5. Update the User's Wallet
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
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)
