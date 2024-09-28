import pandas as pd
import ollama

# Load CSV files
user_data = pd.read_csv('input/hd116119-2.csv', delimiter=";")
opt_data = pd.read_csv('input/optType.csv', delimiter=";")

# Clean up column names (if necessary)
user_data.columns = user_data.columns.str.strip()

# Example function to extract specific user financial data
def get_user_financial_data(user_id):
    user_info = user_data[user_data['ID client'] == user_id]
    total_spent = user_info['suma tranzactiei'].sum()
    frequent_transactions = user_info.groupby('MCC_CODE')['suma tranzactiei'].sum().nlargest(3)

    return total_spent, frequent_transactions

# 1. Anomaly Detection Agent
def anomaly_detection_agent(user_id):
    user_info = user_data[user_data['ID client'] == user_id]
    avg_transaction = user_info['suma tranzactiei'].mean()
    large_transactions = user_info[user_info['suma tranzactiei'] > (2 * avg_transaction)]

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

    return response['message']['content']

# 2. Budget Prediction Agent
def budget_prediction_agent(user_id):
    user_info = user_data[user_data['ID client'] == user_id]
    total_spent = user_info['suma tranzactiei'].sum()
    avg_monthly_spent = total_spent / (user_info['data si ora'].nunique() / 30)  # Approx monthly average

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

    return response['message']['content']

# 3. Recommendation Engine Agent
def recommendation_engine_agent(user_id):
    user_info = user_data[user_data['ID client'] == user_id]
    total_savings = user_info['suma tranzactiei'].sum() * 0.20  # Assume 20% of spending can be saved
    categories = user_info.groupby('MCC_CODE')['suma tranzactiei'].sum().nlargest(3).to_dict()

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

    return response['message']['content']

# 4. Transaction Categorization Agent
def transaction_categorization_agent(user_id):
    user_info = user_data[user_data['ID client'] == user_id]
    categories = user_info.groupby('MCC_CODE')['suma tranzactiei'].sum().to_dict()

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

    return response['message']['content']

# Main function to execute the flow
def main():
    # List all columns to verify the correct column names
    print("Columns in user_data:", user_data.columns)

    # Get unique user IDs
    user_ids = user_data['ID client'].unique()

    # Run agents for each user
    for user_id in user_ids:
        print(f"User ID: {user_id}")
        print("Anomaly Detection:", anomaly_detection_agent(user_id))
        print("Budget Prediction:", budget_prediction_agent(user_id))
        print("Investment Recommendations:", recommendation_engine_agent(user_id))
        print("Transaction Categorization:", transaction_categorization_agent(user_id))
        print("\n")

if __name__ == "__main__":
    main()
