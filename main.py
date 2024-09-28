import pandas as pd
import ollama
from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_agents.agents import anomaly_detection_agent, budget_prediction_agent, recommendation_engine_agent, \
    transaction_categorization_agent, router_agent_with_llama

# Load CSV files
user_data = pd.read_csv('input/hd116119-2.csv', delimiter=";")
opt_data = pd.read_csv('input/optType.csv', delimiter=";")
user_data.columns = user_data.columns.str.strip()

# Check if opt_data is empty
if opt_data.empty:
    print("opt_data is empty, handling accordingly.")

app = Flask("Spendog")
CORS(app)  # This will automatically handle CORS and OPTIONS requests

@app.route('/generate', methods=['OPTIONS', 'POST'])
def generate_response():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    try:
        # Get the prompt and user_id from the POST request
        data = request.get_json()
        prompt = data.get("prompt", "")
        user_id = data.get("user_id", "")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        if not user_id:
            return jsonify({"error": "No user_id provided"}), 400

        # Check if user exists in the dataset
        user_info = user_data[user_data['ID client'] == user_id]
        if user_info.empty:
            return jsonify({"error": "User not found"}), 404

        # Call the Router Agent to determine the correct agent(s)
        agent_response = router_agent_with_llama(prompt, user_info)

        return jsonify({"response": agent_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _build_cors_preflight_response():
    """Helper function to build the preflight response."""
    response = jsonify({"message": "CORS preflight successful"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
