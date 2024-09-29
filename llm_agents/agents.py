import ollama

AVAILABLE_METHODS = {
    "anomaly_detection_agent": "Detects suspicious or unusual transactions in user data.",
    "budget_prediction_agent": "Generates a personalized budget based on user spending patterns.",
    "recommendation_engine_agent": "Provides investment advice based on spending and saving behavior.",
    "transaction_categorization_agent": "Categorizes user transactions for better tracking and analysis."
}

# Helper function to invoke the correct agent based on Llama's output
def call_agent(agent_name, user_info):
    print(f"Calling agent: {agent_name}")
    if agent_name == "anomaly_detection_agent":
        return anomaly_detection_agent(user_info)
    elif agent_name == "budget_prediction_agent":
        return budget_prediction_agent(user_info)
    elif agent_name == "recommendation_engine_agent":
        return recommendation_engine_agent(user_info)
    elif agent_name == "transaction_categorization_agent":
        return transaction_categorization_agent(user_info)
    else:
        print("Invalid agent selected.")
        return "Invalid agent selected."

# Router Agent using Llama 3.2 to classify and choose the correct method
def router_agent_with_llama(prompt, user_info):
    print(f"User Prompt: {prompt}")

    suggested_methods = []

    for method in AVAILABLE_METHODS:
        if method in prompt:
            suggested_methods.append(method)

    if len(suggested_methods) == 0:
        # Convert the list of methods to a readable format for Llama 3.2
        methods_description = "\n".join([f"{method}: {description}" for method, description in AVAILABLE_METHODS.items()])

        # Prepare the prompt for Llama 3.2
        llama_prompt = f"""
        The following are the available methods:
        {methods_description}

        Based on the following user prompt, identify which methods should be used:
        "{prompt}". Please answer shortly with just an array of matching methods
        """
        print(f"Sending the following prompt to Llama 3.2: {llama_prompt}")

        # Query Llama 3.2
        response = ollama.chat(model="llama3.2", messages=[
            {"role": "user", "content": llama_prompt}
        ])

        print(f"Llama 3.2 Response: {response['message']['content']}")

        # Process the response to extract the suggested methods
        suggested_methods = extract_methods_from_response(response['message']['content'])

    print(f"Suggested Methods: {suggested_methods}")

    if len(suggested_methods) == 0:
        return "I apologize, I couldn't understand your request. Could you please rephrase or provide more context? Feel free to ask a different question."

    # Call the matching methods with the user prompt
    results = {}
    for method in suggested_methods:
        if method in AVAILABLE_METHODS:
            print(f"Invoking method: {method}")
            results[method] = call_agent(method, user_info)

    print(f"Final Results: {results}")
    return results

# Helper function to extract method names from Llama's response
def extract_methods_from_response(llama_response):
    # Parse Llama's response to identify matching method names
    print(f"Extracting methods from Llama response: {llama_response}")
    matched_methods = []
    for method in AVAILABLE_METHODS.keys():
        if method in llama_response.lower():
            matched_methods.append(method)
    print(f"Matched Methods: {matched_methods}")
    return matched_methods


# Example function to extract specific user financial data
def get_user_financial_data(user_info):
    total_spent = user_info['suma tranzactiei'].sum()
    frequent_transactions = user_info.groupby('MCC_CODE')['suma tranzactiei'].sum().nlargest(3)

    print(f"Total spent: {total_spent}, Frequent transactions: {frequent_transactions}")
    return total_spent, frequent_transactions

# 1. Anomaly Detection Agent
def anomaly_detection_agent(user_info):
    avg_transaction = user_info['suma tranzactiei'].mean()
    large_transactions = user_info[user_info['suma tranzactiei'] > (2 * avg_transaction)]

    print(f"Average transaction: {avg_transaction}")
    print(f"Large transactions: {large_transactions[['data si ora', 'suma tranzactiei', 'MCC_CODE']]}")

    prompt = f"""
    Analyze the user's transactions to detect anomalies. The average transaction is {avg_transaction:.2f} MDL. 
    Flag the following large or suspicious transactions: {large_transactions[['data si ora', 'suma tranzactiei', 'MCC_CODE']].to_dict()}.
    """

    response = ollama.chat(model='neural-chat', messages=[
        {
            'role': 'user',
            'content': prompt
        },
    ])

    print(f"Anomaly Detection Response: {response['message']['content']}")
    return response['message']['content']

# 2. Budget Prediction Agent
def budget_prediction_agent(user_info):
    total_spent = user_info['suma tranzactiei'].sum()
    avg_monthly_spent = total_spent / (user_info['data si ora'].nunique() / 30)  # Approx monthly average

    print(f"Total spent: {total_spent}, Average monthly spent: {avg_monthly_spent}")

    prompt = f"""
    The user spends approximately {avg_monthly_spent:.2f} MDL per month. Based on their transaction history, 
    generate a personalized monthly budget that takes into account their income and regular expenses.
    """

    response = ollama.chat(model='neural-chat', messages=[
        {
            'role': 'user',
            'content': prompt
        },
    ])

    print(f"Budget Prediction Response: {response['message']['content']}")
    return response['message']['content']

# 3. Recommendation Engine Agent
def recommendation_engine_agent(user_info):
    total_savings = user_info['suma tranzactiei'].sum() * 0.20  # Assume 20% of spending can be saved
    categories = user_info.groupby('MCC_CODE')['suma tranzactiei'].sum().nlargest(3).to_dict()

    print(f"Total savings: {total_savings}, Top categories: {categories}")

    prompt = f"""
    Based on the user's savings potential of {total_savings:.2f} MDL, provide personalized investment advice. 
    Their top spending categories are: {categories}. Recommend low-risk and medium-risk investment options based on their financial behavior.
    """

    response = ollama.chat(model='neural-chat', messages=[
        {
            'role': 'user',
            'content': prompt
        },
    ])

    print(f"Recommendation Response: {response['message']['content']}")
    return response['message']['content']

# 4. Transaction Categorization Agent
def transaction_categorization_agent(user_info):
    categories = user_info.groupby('MCC_CODE')['suma tranzactiei'].sum().to_dict()

    print(f"Transaction categories: {categories}")

    prompt = f"""
    Categorize the user's transactions based on the MCC codes. The following are the total amounts spent in each category: {categories}.
    Generate a summary of the user's expenses and suggest ways to optimize spending.
    """

    response = ollama.chat(model='neural-chat', messages=[
        {
            'role': 'user',
            'content': prompt
        },
    ])

    print(f"Transaction Categorization Response: {response['message']['content']}")
    return response['message']['content']
