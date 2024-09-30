import asyncio
import ollama
from typing import List

import pandas as pd

AVAILABLE_METHODS = {
    "anomaly_detection_agent": "Detects suspicious or unusual transactions in user data.",
    "budget_prediction_agent": "Generates a personalized budget based on user spending patterns.",
    "recommendation_engine_agent": "Provides investment advice based on spending and saving behavior.",
    "transaction_categorization_agent": "Categorizes user transactions for better tracking and analysis."
}

# Helper function to invoke the correct agent based on Llama's output
async def call_agent(agent_name, user_info, prompt=None):
    print(f"Calling agent: {agent_name}")
    if agent_name == "anomaly_detection_agent":
        return await anomaly_detection_agent(user_info)
    elif agent_name == "budget_prediction_agent":
        return await budget_prediction_agent(user_info)
    elif agent_name == "recommendation_engine_agent":
        return await recommendation_engine_agent(user_info)
    elif agent_name == "transaction_categorization_agent":
        return await transaction_categorization_agent(user_info)
    elif agent_name == "fallback_open_prompt_agent":
        return await fallback_open_prompt_agent(user_info, prompt)
    else:
        return {"response": "Invalid agent selected."}

# Asynchronous Router Agent using Llama 3.2 to classify and choose the correct method
async def router_agent_with_llama(prompt, user_info):
    print(f"User Prompt: {prompt}")

    # Convert the list of methods to a readable format for Llama 3.2
    methods_description = "\n".join([f"{method}: {description}" for method, description in AVAILABLE_METHODS.items()])

    # Prepare the prompt for Llama 3.2
    llama_prompt = f"""
    The following are the available methods:
    {methods_description}
    
    Based on the following user prompt, identify which methods should be used:
    "{prompt}". Please answer shortly with just an array of matching methods.
    """
    print(f"Sending the following prompt to Llama 3.2: {llama_prompt}")

    # Since ollama.chat is synchronous, use asyncio.to_thread to run it asynchronously
    response = await asyncio.to_thread(
        ollama.chat,
        model="llama3.2",
        messages=[{"role": "user", "content": llama_prompt}]
    )

    llama_response_content = response['message']['content']
    print(f"Llama 3.2 Response: {llama_response_content}")

    # Process the response to extract the suggested methods
    suggested_methods = extract_methods_from_response(llama_response_content)

    print(f"Suggested Methods: {suggested_methods}")

    # If no matching methods, redirect to the fallback agent
    if not suggested_methods:
        print(f"Redirecting to fallback_open_prompt_agent")
        return await call_agent("fallback_open_prompt_agent", user_info, prompt)

    # Call the matching methods concurrently
    tasks = [call_agent(method, user_info) for method in suggested_methods]
    results = await asyncio.gather(*tasks)

    print(f"Final Results: {results}")
    return dict(zip(suggested_methods, results))

# Helper function to extract method names from Llama's response
def extract_methods_from_response(llama_response: str) -> List[str]:
    print(f"Extracting methods from Llama response: {llama_response}")
    matched_methods = []
    for method in AVAILABLE_METHODS.keys():
        if method in llama_response.lower():
            matched_methods.append(method)
    print(f"Matched Methods: {matched_methods}")
    return matched_methods

# Fallback Agent for Open-ended Prompts
async def fallback_open_prompt_agent(user_info, user_prompt):
    # Prepare a simple prompt that contains all user data and the user's question
    prompt = f"""
    The user has provided the following transaction data in tabular format:
    {user_info.to_dict()}
    
    Based on this data, answer the following prompt from the user in markdown format:
    {user_prompt}
    """

    # Send the prompt to Ollama as an open-ended question
    response = await asyncio.to_thread(
        ollama.chat,
        model='neural-chat',
        messages=[{'role': 'user', 'content': prompt}]
    )

    print(f"Fallback Agent Response: {response['message']['content']}")
    return {"response": response['message']['content']}

# 1. Anomaly Detection Agent
async def anomaly_detection_agent(user_info):
    # Calculate the average transaction and set a threshold for significant anomalies
    avg_transaction = user_info['suma tranzactiei'].mean()
    threshold = 2 * avg_transaction  # We use a 2x average as the anomaly threshold
    significant_anomalies = user_info[user_info['suma tranzactiei'] > threshold]

    # Check if there are any significant anomalies
    if significant_anomalies.empty:
        return "No significant anomalies detected."

    # Extract relevant transaction details and prepare markdown-friendly format
    transactions_info = []
    for index, row in significant_anomalies.iterrows():
        transaction_info = f"- **Transaction ID**: {index}, **MCC Code**: {row['MCC_CODE']}, **Date and Time**: {row['data si ora']}, **Amount**: {row['suma tranzactiei']:.2f} MDL"
        transactions_info.append(transaction_info)

    # Join the transaction info into a single string to pass to the prompt
    transactions_markdown = "\n".join(transactions_info)

    # Generate a prompt that focuses only on the significant anomalies
    prompt = f"""
        Let's focus on the significant anomalies in MDL currency. The threshold for detecting anomalies is {threshold:.2f} MDL.
        Please analyze only the following transactions and explain why each is flagged as an anomaly:
        {transactions_markdown}
        For each transaction, explain briefly why it is flagged as an anomaly in markdown format.
        """

    # Use asyncio.to_thread to run the synchronous ollama.chat asynchronously
    response = await asyncio.to_thread(
        ollama.chat,
        model='neural-chat',
        messages=[{'role': 'user', 'content': prompt}]
    )

    # Output the response from the Llama model
    print(f"Anomaly Detection Response: {response['message']['content']}")
    return response['message']['content']

# 2. Budget Prediction Agent
async def budget_prediction_agent(user_info):
    savings_goal = 10000
    months_remaining = 12
    # Handle missing or incorrect data gracefully
    if 'suma tranzactiei' not in user_info.columns or 'data si ora' not in user_info.columns or 'MCC_CODE' not in user_info.columns:
        return {"response": "Invalid data format. Columns 'suma tranzactiei', 'data si ora', or 'MCC_CODE' are missing."}

    # Convert 'data si ora' to datetime
    if not pd.api.types.is_datetime64_any_dtype(user_info['data si ora']):
        user_info['data si ora'] = pd.to_datetime(user_info['data si ora'], errors='coerce')

    # Filter out invalid dates
    user_info = user_info.dropna(subset=['data si ora'])

    # Calculate total spending
    total_spent = user_info['suma tranzactiei'].sum()

    # Calculate the number of months spanned by the transaction data
    start_date = user_info['data si ora'].min()
    end_date = user_info['data si ora'].max()

    # Calculate the total number of days and approximate months
    days_spanned = (end_date - start_date).days
    num_months = days_spanned / 30.44  # Approximate number of months based on average days per month

    if num_months <= 0:
        return {"response": "Insufficient data to calculate monthly spending."}

    # Calculate average monthly spending
    avg_monthly_spent = total_spent / num_months

    # Categorize transactions by 'MCC_CODE'
    category_spending = user_info.groupby('MCC_CODE')['suma tranzactiei'].sum().sort_values(ascending=False)

    # Get the top category to cut
    top_category_to_cut = category_spending.idxmax()
    spending_in_top_category = category_spending.max()

    # Calculate the monthly savings target to reach the goal
    monthly_savings_target = savings_goal / months_remaining

    # Calculate potential savings from the top category (suggest a 20-30% cut to gradually meet the target)
    potential_savings = spending_in_top_category * 0.25  # Suggest cutting 25% from the top category

    # Formulate a concise prompt for Ollama
    prompt = f"""
    The user needs to save 10,000 LEI over the next {months_remaining} months. This requires approximately {monthly_savings_target:.2f} LEI per month in savings.
    The biggest spending category is MCC Code '{top_category_to_cut}', accounting for {spending_in_top_category:.2f} LEI.
    Suggest how the user can gradually reduce spending in this category or other categories to meet their savings goal. Please provide the response in markdown format.
    """

    # Use asyncio.to_thread to run the synchronous ollama.chat asynchronously
    response = await asyncio.to_thread(
        ollama.chat,
        model='neural-chat',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']

# 3. Recommendation Engine Agent
async def recommendation_engine_agent(user_info):
    # Define constants
    annual_rate = 0.10
    compounding_periods = 12
    years_list = [5, 10, 20]
    total_income = 10000  # Assume a fixed monthly income, or fetch from user data

    # Handle missing or incorrect data gracefully
    if 'suma tranzactiei' not in user_info.columns or 'data si ora' not in user_info.columns:
        return {"response": "Invalid data format. Columns 'suma tranzactiei' or 'data si ora' are missing."}

    # Convert 'data si ora' to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(user_info['data si ora']):
        user_info['data si ora'] = pd.to_datetime(user_info['data si ora'], errors='coerce')

    # Filter out invalid dates
    user_info = user_info.dropna(subset=['data si ora'])

    # Calculate total spending over the last 12 months
    last_12_months_data = user_info[user_info['data si ora'] > (user_info['data si ora'].max() - pd.DateOffset(months=12))]
    total_spent = last_12_months_data['suma tranzactiei'].sum()

    # Calculate average monthly savings
    avg_monthly_savings = total_income - (total_spent / 12)

    if avg_monthly_savings <= 0:
        return {"response": "The user does not have enough savings to invest based on the last 12 months of data."}

    # Helper function to calculate future value
    def calculate_future_value(monthly_savings, years, annual_rate=0.10, compounding_periods=12):
        total_months = years * compounding_periods
        future_value = monthly_savings * (((1 + (annual_rate / compounding_periods)) ** total_months - 1) / (annual_rate / compounding_periods))
        return future_value

    # Calculate future values for 5, 10, and 20 years
    projections = {years: calculate_future_value(avg_monthly_savings, years, annual_rate) for years in years_list}

    # Formulate a concise prompt for Ollama
    prompt = f"""
    Based on the user’s last 12 months of spending, their average monthly savings is approximately {avg_monthly_savings:.2f} LEI.
    Calculate the projected value of these savings if invested at a 10% annual return for 5, 10, and 20 years.
    For 5 years: {projections[5]:.2f} LEI, for 10 years: {projections[10]:.2f} LEI, and for 20 years: {projections[20]:.2f} LEI.
    Provide recommendations for investment strategies and suggest how to maximize returns in markdown format.
    """

    # Use asyncio.to_thread to run the synchronous ollama.chat asynchronously
    response = await asyncio.to_thread(
        ollama.chat,
        model='neural-chat',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']

# 4. Transaction Categorization Agent
async def transaction_categorization_agent(user_info):
    # Group transactions by MCC_CODE and calculate total spending per category
    categories = user_info.groupby('MCC_CODE')['suma tranzactiei'].sum().to_dict()

    # Prepare the prompt for Ollama to categorize transactions and provide suggestions
    prompt = f"""
    Based on the user's transaction data, categorize the expenses into groups such as groceries, dining, transportation, and utilities. 
    The following are the total amounts spent in each category based on MCC codes: {categories}.
    Highlight which category the user spends the most on, and suggest ways to optimize their spending.
    Provide insights into the top three areas where the user can reduce spending, and how much should be saved in each category in markdown format.
    Also, summarize the user's spending habits in the last 12 months, and provide any suggestions for adjustments.
    """

    # Use asyncio.to_thread to run the synchronous ollama.chat asynchronously
    response = await asyncio.to_thread(
        ollama.chat,
        model='neural-chat',
        messages=[{'role': 'user', 'content': prompt}]
    )

    # Print and return the response for further use in the application
    print(f"Transaction Categorization Response: {response['message']['content']}")
    return response['message']['content']
