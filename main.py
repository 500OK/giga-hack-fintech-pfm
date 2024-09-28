import pandas as pd
from flask import Flask, request, jsonify
import ollama
from llm_agents.agents import anomaly_detection_agent, budget_prediction_agent, recommendation_engine_agent, \
    transaction_categorization_agent

# Load CSV files
user_data = pd.read_csv('input/hd116119-2.csv', delimiter=";")
opt_data = pd.read_csv('input/optType.csv', delimiter=";")

# Clean up column names (if necessary)
user_data.columns = user_data.columns.str.strip()

app = Flask("Spendog")

# Agent dispatcher function
def agent_dispatcher(prompt, user_info):
    if 'anomaly' in prompt.lower():
        return anomaly_detection_agent(user_info)
    elif 'budget' in prompt.lower():
        return budget_prediction_agent(user_info)
    elif 'investment' in prompt.lower() or 'recommend' in prompt.lower():
        return recommendation_engine_agent(user_info)
    elif 'categorize' in prompt.lower() or 'transaction' in prompt.lower():
        return transaction_categorization_agent(user_info)
    else:
        return "Sorry, I didn't understand the request. Please ask about anomaly detection, budgeting, investment advice, or transaction categorization."

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get the prompt from the POST request
        data = request.get_json()
        prompt = data.get("prompt", "")
        user_id = data.get("user_id", "")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        if not user_id:
            return jsonify({"error": "No user_id provided"}), 400

        # Find user in the dataset
        user_info = user_data[user_data['ID client'] == user_id]
        if user_info.empty:
            return jsonify({"error": "User not found"}), 404

        # Dispatch to the correct agent
        agent_response = agent_dispatcher(prompt, user_info)

        return jsonify({"response": agent_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
