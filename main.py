import pandas as pd
import ollama
from flask import Flask, request, jsonify
from llm_agents.agents import anomaly_detection_agent, budget_prediction_agent, recommendation_engine_agent, \
    transaction_categorization_agent, router_agent_with_llama

# Load CSV files
user_data = pd.read_csv('input/hd116119-2.csv', delimiter=";")
opt_data = pd.read_csv('input/optType.csv', delimiter=";")
user_data.columns = user_data.columns.str.strip()

app = Flask("Spendog")


@app.route('/generate', methods=['POST'])
def generate_response():
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

if __name__ == "__main__":
    app.run(debug=True)
