import streamlit as st
import pandas as pd
import json
import openai
import os
import re
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType


# Setting up the api key
import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")

def csv_agent_func(file_path, user_message):
    """Run the CSV agent with the given file path and user message."""
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4-turbo", openai_api_key=API_KEY),
        file_path, 
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    try:
        # Properly format the user's input and wrap it with the required "input" key
        tool_input = {
            "input": {
                "name": "python",
                "arguments": user_message
            }
        }
        
        response = agent.run(tool_input)
        return response
    except Exception as e:
        st.write(f"Error: {e}")
        return None

def display_content_from_json(json_response):
    """
    Display content to Streamlit based on the structure of the provided JSON.
    """
    
    # Check if the response has plain text.
    if "answer" in json_response:
        st.write(json_response["answer"])

    # Check if the response has a bar chart.
    if "bar" in json_response:
        data = json_response["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response has a table.
    if "table" in json_response:
        data = json_response["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def extract_code_from_response(response):
    """Extracts Python code from a string response."""
    # Use a regex pattern to match content between triple backticks
    code_pattern = r"```python(.*?)```"
    match = re.search(code_pattern, response, re.DOTALL)
    
    if match:
        # Extract the matched code and strip any leading/trailing whitespaces
        return match.group(1).strip()
    return None

